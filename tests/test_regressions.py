import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ['LITELLM_LOCAL_MODEL_COST_MAP']='True'
import pytest
from scratch_agents import Agent, ExecutionContext, LlmResponse, Message, ToolCall, ToolResult, Event
from scratch_agents.planning import create_tasks
from scratch_agents.workflows.parallel import ParallelWorkflow
from scratch_agents.rag import fixed_length_chunking

ROOT=Path(__file__).resolve().parents[1]

def cell(ch,index):
    return ''.join(json.loads(next((ROOT/'notebooks'/ch).glob('*.ipynb')).read_text())['cells'][index-1]['source'])

class Model:
    model="offline-test"
    def __init__(self,text='done',error=None):self.text=text;self.error=error
    async def generate(self,request):
        await asyncio.sleep(0)
        return LlmResponse(error_message=self.error,content=[] if self.error else [Message(role='assistant',content=self.text)])

@pytest.mark.parametrize('cls',[ParallelWorkflow,'notebook'])
def test_parallel_branches_do_not_share_or_duplicate_context(cls):
    if cls=='notebook':
        ns={};exec(cell('ch09',23),ns);cls=ns['ParallelWorkflow']
    class Worker:
        def __init__(self,name):self.name=name
        async def run(self,user_input,context,**kw):
            from scratch_agents import AgentResult
            assert context.current_step==0 and context.final_result is None
            context.state['nested']['owner']=self.name
            context.add_event(Event(execution_id=context.execution_id,author='user',content=[Message(role='user',content=user_input)]))
            await asyncio.sleep(0)
            assert context.state['nested']['owner']==self.name
            context.add_event(Event(execution_id=context.execution_id,author=self.name,content=[Message(role='assistant',content=self.name)]))
            return AgentResult(self.name,context)
    seed=ExecutionContext(state={'nested':{'owner':'seed'}},current_step=9,final_result='previous')
    seed.add_event(Event(execution_id=seed.execution_id,author='user',content=[Message(role='user',content='old')]))
    result=asyncio.run(cls([Worker('a'),Worker('b')]).run('new',context=seed))
    assert len(seed.events)==1 and seed.state['nested']['owner']=='seed'
    assert len(result.context.events)==4
    assert [e.author for e in result.context.events]==['user','user','a','b']
    assert result.output=='[a]\na\n\n[b]\nb'

def test_llm_error_is_not_reported_as_success():
    with pytest.raises(RuntimeError,match='LLM request failed: offline failure'):
        asyncio.run(Agent(Model(error='offline failure')).run('hi'))

class Sandbox:
    def __init__(self):self.closed=False;self.uploads={};self.files=SimpleNamespace(write=lambda path,data:self.uploads.update({path:data}))
    def run_code(self,code):return SimpleNamespace(error=None)
    def kill(self):self.closed=True

def test_sandbox_cleanup_and_skill_directory_upload(tmp_path):
    skill=tmp_path/'example';skill.mkdir();(skill/'SKILL.md').write_text('---\nname: example\ndescription: Test skill\n---\nTest')
    (skill/'helper.py').write_text('print(1)')
    sandboxes=[]
    def create(**kw):
        s=Sandbox();sandboxes.append(s);return s
    import e2b_code_interpreter
    with patch.object(e2b_code_interpreter.Sandbox,'create',side_effect=create):
        agent=Agent(Model(),code_execution='e2b',skills_path=str(tmp_path))
        result=asyncio.run(agent.run('hi'))
        assert result.context.code_env is None and sandboxes[0].closed
        assert sandboxes[0].uploads['/home/user/skills/example/helper.py']==b'print(1)'
        result.context.final_result=None;result.context.current_step=0
        asyncio.run(agent.run('again',context=result.context))
        assert len(sandboxes)==2 and sandboxes[1].closed

def test_sandbox_setup_failure_is_visible():
    import e2b_code_interpreter
    with patch.object(e2b_code_interpreter.Sandbox,'create',side_effect=RuntimeError('offline setup failure')):
        with pytest.raises(RuntimeError,match='Failed to set up'):
            asyncio.run(Agent(Model(),code_execution='e2b').run('hi'))

@pytest.mark.parametrize('target',['package','notebook'])
def test_planning_accepts_json_tool_arguments(target):
    if target=='notebook':
        ns={};exec(cell('ch07',6),ns);exec(cell('ch07',9),ns);fn=ns['create_tasks']
    else:fn=create_tasks
    result=asyncio.run(fn(ExecutionContext(),tasks=[{'content':'Find evidence','status':'pending'}]))
    assert result=='[ ] Find evidence'

@pytest.mark.parametrize('target',['package','notebook'])
def test_chunker_rejects_nonadvancing_window(target):
    if target=='notebook':
        ns={};exec(cell('ch05',11),ns);fn=ns['fixed_length_chunking']
    else:fn=fixed_length_chunking
    with pytest.raises(ValueError):fn('abc',chunk_size=1,overlap=1)
    assert fn('abcdef',chunk_size=3,overlap=1)==['abc','cde','ef']

def test_notebook_csv_excel_readers(tmp_path):
    ns={};exec(cell('ch05',49),ns)
    import pandas as pd
    frame=pd.DataFrame({'value':[42]})
    csv=tmp_path/'data.csv';xlsx=tmp_path/'data.xlsx'
    frame.to_csv(csv,index=False);frame.to_excel(xlsx,index=False)
    assert '42' in ns['_read_csv'](csv)
    assert '42' in ns['_read_excel'](xlsx)

def test_notebook_token_count_and_callbacks():
    from scratch_agents import LlmRequest,LlmClient
    from typing import Optional
    ns={'LlmRequest':LlmRequest,'ExecutionContext':ExecutionContext,'LlmResponse':LlmResponse,'Optional':Optional,'json':json}
    exec(cell('ch06',15),ns)
    encoding=SimpleNamespace(encode=lambda s:list(s.encode()))
    with patch('tiktoken.encoding_for_model',return_value=encoding):
        assert ns['count_tokens'](LlmRequest(contents=[Message(role='user',content='hi')]))==6


def test_notebook_command_formatter_matches_e2b_result():
    from scratch_agents.tools import tool
    ns={'tool':tool,'ExecutionContext':ExecutionContext}
    exec(cell('ch08',61),ns)
    # Exercise the installed SDK result type rather than an invented shape.
    from e2b.sandbox.commands.command_handle import CommandResult
    result=CommandResult(stdout='hello',stderr='',exit_code=0,error='')
    assert 'hello' in ns['_format_command_result'](result)

def test_notebook_failing_tool_is_recorded_and_agent_can_continue():
    from scratch_agents.tools import tool
    ns={'tool':tool};exec(cell('ch07',18),ns)
    class RecoveryModel:
        model='offline-test'
        calls=0
        async def generate(self,request):
            self.calls+=1
            if self.calls==1:
                return LlmResponse(content=[ToolCall(tool_call_id='failed-wiki',name='get_wikipedia_page',arguments={'title':'Moon'})])
            assert any(isinstance(c,ToolResult) and c.status=='error' for c in request.contents)
            return LlmResponse(content=[Message(role='assistant',content='Use another source')])
    result=asyncio.run(Agent(RecoveryModel(),tools=[ns['get_wikipedia_page']]).run('Research the Moon'))
    assert result.output=='Use another source'

@pytest.mark.parametrize('target', ['package', 'notebook'])
@pytest.mark.parametrize('status', ['pending_confirmation', 'error'])
def test_parallel_keeps_incomplete_branch_results(target, status):
    from scratch_agents import AgentResult
    from scratch_agents.workflows.parallel import ParallelWorkflowIncomplete
    workflow = ParallelWorkflow
    error_type = ParallelWorkflowIncomplete
    if target == 'notebook':
        ns = {}; exec(cell('ch09', 23), ns)
        workflow = ns['ParallelWorkflow']; error_type = ns['ParallelWorkflowIncomplete']
    pending_context = ExecutionContext(state={'pending_tool_calls': ['keep-me']})
    partial = AgentResult(None, pending_context, status=status, pending_tool_calls=['approval'])
    class Worker:
        name = 'waiting'
        async def run(self, *args, **kwargs): return partial
    with pytest.raises(error_type, match='did not complete') as raised:
        asyncio.run(workflow([Worker()]).run('test'))
    assert raised.value.branch_results[0] == ('waiting', partial)
    assert raised.value.branch_results[0][1].context is pending_context
    assert raised.value.branch_results[0][1].pending_tool_calls == ['approval']

@pytest.fixture
def deletion_demo():
    ns = {'model': Model()}
    exec(cell('ch06', 67), ns)
    try: yield ns
    finally: ns['demo_directory'].cleanup()

@pytest.mark.parametrize('attack', ['other_file', 'symlink', 'wrong_tool', 'extra_argument'])
def test_demo_approval_rejects_unexpected_request(deletion_demo, tmp_path, attack):
    from scratch_agents.context import PendingToolCall
    ns = deletion_demo
    victim = tmp_path/'unrelated.txt'; victim.write_text('keep')
    arguments = {'filename': str(victim)}; name = 'delete_file'
    if attack == 'symlink':
        link = tmp_path/'link.txt'
        try: link.symlink_to(ns['demo_file'])
        except OSError: pytest.skip('This OS does not permit symlink creation')
        arguments = {'filename': str(link)}
    elif attack == 'wrong_tool':
        arguments = {'filename': str(ns['demo_file'])}; name = 'other_tool'
    elif attack == 'extra_argument':
        arguments = {'filename': str(ns['demo_file']), 'other': 'unexpected'}
    pending = PendingToolCall(tool_call=ToolCall(tool_call_id='bad', name=name, arguments=arguments), confirmation_message='test')
    with pytest.raises((PermissionError, ValueError)):
        exec(cell('ch06', 71).split('# User approves')[0], {**ns, 'result': SimpleNamespace(pending_tool_calls=[pending])})
    assert victim.read_text() == 'keep' and ns['demo_file'].exists()

@pytest.mark.parametrize('symlink', [False, True])
def test_demo_tool_blocks_wrong_path_even_without_approval_check(deletion_demo, tmp_path, symlink):
    ns = deletion_demo
    victim = tmp_path/'unrelated.txt'; victim.write_text('keep')
    requested = victim
    if symlink:
        requested = tmp_path/'alias'
        try: requested.symlink_to(ns['demo_file'])
        except OSError: pytest.skip('This OS does not permit symlink creation')
    fn = ns['delete_file']
    with pytest.raises(PermissionError):
        asyncio.run(fn(ExecutionContext(), filename=str(requested)))
    assert victim.read_text() == 'keep' and ns['demo_file'].exists()

def test_demo_approved_file_is_deleted(deletion_demo):
    import ast
    ns = deletion_demo
    class DeleteModel(Model):
        count = 0
        async def generate(self, request):
            self.count += 1
            if self.count == 1:
                return LlmResponse(content=[ToolCall(tool_call_id='demo', name='delete_file', arguments={'filename':str(ns['demo_file'])})])
            return LlmResponse(content=[Message(role='assistant', content='Deleted')])
    ns['agent'].model = DeleteModel()
    async def run():
        for i in (69, 71):
            await eval(compile(cell('ch06', i), '<notebook>', 'exec', flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT), ns)
    asyncio.run(run())
    assert not ns['demo_file'].exists()

def test_parallel_pending_branch_can_be_resumed_without_rerunning_workflow():
    from scratch_agents import ToolConfirmation
    from scratch_agents.tools import tool
    from scratch_agents.workflows.parallel import ParallelWorkflowIncomplete
    actions = []
    @tool(requires_confirmation=True)
    def record(value: str):
        actions.append(value)
        return value
    class ConfirmationModel(Model):
        async def generate(self, request):
            if any(isinstance(c, ToolResult) for c in request.contents):
                return LlmResponse(content=[Message(role='assistant', content='recorded')])
            return LlmResponse(content=[ToolCall(tool_call_id='confirm-record', name='record', arguments={'value':'once'})])
    worker = Agent(ConfirmationModel(), tools=[record], name='worker')
    async def run():
        with pytest.raises(ParallelWorkflowIncomplete) as raised:
            await ParallelWorkflow([worker]).run('record once')
        name, pending = raised.value.branch_results[0]
        assert name == 'worker' and not actions
        confirmation = ToolConfirmation(tool_call_id=pending.pending_tool_calls[0].tool_call.tool_call_id, approved=True)
        result = await worker.run(context=pending.context, tool_confirmations=[confirmation])
        assert result.status == 'complete' and result.output == 'recorded'
    asyncio.run(run())
    assert actions == ['once']

@pytest.mark.parametrize('target', ['package', 'notebook'])
def test_parallel_preserves_results_when_an_agent_raises(target):
    from scratch_agents.workflows.parallel import ParallelWorkflowIncomplete
    workflow = ParallelWorkflow
    if target == 'notebook':
        ns = {}; exec(cell('ch09', 23), ns); workflow = ns['ParallelWorkflow']
    from scratch_agents import AgentResult
    contexts = {}
    failure = RuntimeError('provider unavailable')
    class Worker:
        def __init__(self, name): self.name = name
        async def run(self, user_input, context, **kwargs):
            contexts[self.name] = context
            if self.name == 'failed': raise failure
            await asyncio.sleep(0.01)
            return AgentResult(self.name, context, status='pending_confirmation' if self.name == 'waiting' else 'complete')
    with pytest.raises(ParallelWorkflowIncomplete) as raised:
        asyncio.run(workflow([Worker('done'), Worker('waiting'), Worker('failed')]).run('test'))
    assert [r.status for _, r in raised.value.branch_results] == ['complete', 'pending_confirmation', 'error']
    assert all(r.context is contexts[name] for name, r in raised.value.branch_results)
    assert raised.value.branch_errors == (('failed', failure),)

@pytest.mark.parametrize('target', ['package', 'notebook'])
def test_parallel_cancellation_is_not_a_normal_result(target):
    workflow = ParallelWorkflow
    if target == 'notebook':
        ns = {}; exec(cell('ch09', 23), ns); workflow = ns['ParallelWorkflow']
    async def scenario():
        started = asyncio.Event(); cleaned = []
        class Worker:
            name = 'worker'
            async def run(self, *args, **kwargs):
                try:
                    started.set()
                    await asyncio.Event().wait()
                finally: cleaned.append(True)
        task = asyncio.create_task(workflow([Worker()]).run('test'))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError): await task
        assert cleaned == [True]
    asyncio.run(scenario())

def test_restored_transfer_factory_routes_once():
    from typing import List
    ns = {'List': List, 'Agent': Agent, 'ExecutionContext': ExecutionContext}
    exec(cell('ch09', 47), ns)
    transfer = ns['create_transfer_tool']([Agent(Model(), name='writer')])
    context = ExecutionContext()
    assert asyncio.run(transfer(context, agent_name='writer')) == 'Transferring to writer...'
    assert context.transfer_to == 'writer'
    assert 'already requested' in asyncio.run(transfer(context, agent_name='writer'))
