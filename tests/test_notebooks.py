"""Execute every code cell in a fresh Jupyter kernel with external services simulated."""
import ast
import json
import os
from pathlib import Path
import shutil
import sys

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
import pytest

ROOT=Path(__file__).resolve().parents[1]
NOTEBOOKS=sorted((ROOT/'notebooks').glob('**/*.ipynb'))

@pytest.mark.parametrize('path',NOTEBOOKS,ids=lambda p:p.parent.name)
def test_run_all(path,tmp_path):
    run_notebook(path, tmp_path)

def run_notebook(path, tmp_path, broken_tool=None):
    # Match the reader's nested working directory, isolate all generated files.
    checkout=tmp_path/'checkout';chapter=checkout/path.parent.relative_to(ROOT)
    chapter.mkdir(parents=True)
    (checkout/'pyproject.toml').write_text('[project]\nname="notebook-test"\n')
    shutil.copytree(ROOT/'scratch_agents', checkout/'scratch_agents', ignore=shutil.ignore_patterns('__pycache__'))
    if broken_tool:
        module = checkout/'scratch_agents/tools/code_execution.py'
        tree = ast.parse(module.read_text())
        target = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == broken_tool)
        target.body = ast.parse(f'raise RuntimeError("INTENTIONALLY BROKEN {broken_tool}")').body
        module.write_text(ast.unparse(ast.fix_missing_locations(tree)))

    spec=tmp_path/'kernels'/'test-python';spec.mkdir(parents=True)
    (spec/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'Notebook regression test','language':'python'}))
    manager=KernelManager(kernel_name='test-python',kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(spec.parent)]))
    nb=nbformat.read(path,as_version=4)
    if path.parent.name == 'ch08':
        # Explicit test-only activation of the book's optional live examples.
        import pandas as pd
        fixture = checkout/'notebooks/ch05/gaia_workspace/7cc4acfa-63fd-4acc-a1a1-e8e529e0a97f.xlsx'
        fixture.parent.mkdir(parents=True)
        pd.DataFrame({'city':['Wharvton','Wharvton','Algrimand'], 'sales':[100,150,200]}).to_excel(fixture,index=False)
        for index in (34,54,65):
            nb.cells[index].source = '\n'.join(
                line[2:] if line.startswith(('# result =', '# print(', '# if not spreadsheet.', '#     raise FileNotFoundError')) else line
                for line in nb.cells[index].source.splitlines())
    bootstrap=ROOT/'tests'/'notebook_services.py'
    nb.cells.insert(0,nbformat.v4.new_code_cell(f'exec(compile(open({str(bootstrap)!r}).read(), {str(bootstrap)!r}, "exec"))'))
    checks = {
        'ch02': "assert all('error' not in row for rows in results.values() for row in rows)",
        'ch04': "assert result.output is not None\nassert all('error' not in row for rows in results.values() for row in rows)",
        'ch05': "assert response.output is not None\nassert agent._setup_tools is not None",
        'ch06': "assert result.status == 'complete'\nassert count_tokens(LlmRequest(contents=[Message(role='user', content='hi')])) > 0",
        'ch07': "assert result.output is not None",
        'ch08': "assert result.output == 'Wharvton: 250'\nassert result.context.code_env is None\nassert any(e['kind'] == 'upload' and e['bytes'] > 0 for e in SERVICE_EVENTS)\nassert any(e['kind'] == 'command' for e in SERVICE_EVENTS)\nassert any(e['kind'] == 'wikipedia_request' for e in SERVICE_EVENTS)\nassert any(e['kind'] == 'code' and e['text'] == '354224848179261915075' for e in SERVICE_EVENTS)\nassert {'execute_python', 'upload_file', 'bash_tool'} <= {e['name'] for e in SERVICE_EVENTS if e['kind'] == 'tool_request'}",

    }
    if path.parent.name in checks:
        nb.cells.append(nbformat.v4.new_code_cell(checks[path.parent.name]))
    client=NotebookClient(nb,km=manager,timeout=90,allow_errors=False,resources={'metadata':{'path':str(chapter)}})
    try:
        client.execute(cwd=str(chapter))
    finally:
        # Save evidence even on failure; pytest's tmp_path location is in its output.
        nbformat.write(nb,tmp_path/f'{path.parent.name}-executed.ipynb')
    errors=[o for c in nb.cells if c.cell_type=='code' for o in c.outputs if o.output_type=='error']
    assert not errors
    assert all(c.execution_count is not None for c in nb.cells if c.cell_type=='code')


@pytest.mark.parametrize('broken_tool', ['execute_python', 'upload_file', 'bash_tool'])
def test_ch08_detects_broken_tools(broken_tool, tmp_path):
    from nbclient.exceptions import CellExecutionError
    path = next(p for p in NOTEBOOKS if p.parent.name == 'ch08')
    with pytest.raises(CellExecutionError, match=f'INTENTIONALLY BROKEN {broken_tool}'):
        run_notebook(path, tmp_path, broken_tool=broken_tool)
