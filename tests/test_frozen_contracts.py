import ast,importlib.util,json
from pathlib import Path
import pytest
from april.config import GenerationConfig,require_frozen_config,SNAPSHOT_PATH

ROOT=Path(__file__).resolve().parents[1]

def script(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/f'{name}.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def test_config_tamper_rejected(tmp_path):
    p=tmp_path/'config.json';d=json.loads(SNAPSHOT_PATH.read_text());d['sections']['april_main']['K']=[1];p.write_text(json.dumps(d))
    with pytest.raises(ValueError):require_frozen_config(p)
    with pytest.raises(ValueError):GenerationConfig(max_new_tokens=257)

def test_table2_complete_populations_and_published_numbers():
    rows=script('regenerate_table2').regenerate()
    assert len(rows)==40
    overall={r['method']:r for r in rows if r['task']=='OVERALL'}
    assert {k:round(v['tps'],2) for k,v in overall.items()}=={'AR':20.62,'LayerSkip':28.04,'DEL':31.53,'PLD':37.25,'APRIL':40.04}
    assert all(v['requests']==5416 for v in overall.values())

def test_table4_paired_populations():
    rows=script('regenerate_table4').regenerate()
    assert len(rows)==32
    values={r['method'][0]:round(r['tps'],2) for r in rows if r['task']=='OVERALL'}
    assert values=={'A':31.48,'B':32.08,'C':40.00,'D':39.45}

def test_no_del_import_or_unsafe_deserialization():
    for p in list((ROOT/'april').glob('*.py'))+list((ROOT/'scripts').glob('*.py')):
        tree=ast.parse(p.read_text())
        for n in ast.walk(tree):
            if isinstance(n,ast.Import):assert all(a.name.split('.')[0].lower()!='del' for a in n.names)
            if isinstance(n,ast.ImportFrom):assert (n.module or '').split('.')[0].lower()!='del'
            if isinstance(n,ast.Call):
                text=ast.unparse(n.func)
                assert text not in ['eval','exec','pickle.load','pickle.loads','torch.load','os.system']
                assert not any(k.arg in ['shell','trust_remote_code'] and isinstance(k.value,ast.Constant) and k.value.value is True for k in n.keywords)

def test_model_load_is_local_safetensors_only():
    tree=ast.parse((ROOT/'april/model_adapter.py').read_text())
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and ast.unparse(n.func).endswith('AutoModelForCausalLM.from_pretrained')]
    assert len(calls)==1
    flags={k.arg:ast.literal_eval(k.value) for k in calls[0].keywords if k.arg in ['local_files_only','use_safetensors','trust_remote_code']}
    assert flags=={'local_files_only':True,'use_safetensors':True,'trust_remote_code':False}

def test_dataset_hash_change_fails_closed(tmp_path,monkeypatch):
    m=script('prepare_data');raw=tmp_path/'raw';raw.mkdir()
    (raw/'humaneval_test.jsonl').write_text(json.dumps({'task_id':'HumanEval/0','prompt':'changed','canonical_solution':'changed'})+'\n')
    out=tmp_path/'prepared'
    monkeypatch.setattr('sys.argv',['prepare_data','--config',str(SNAPSHOT_PATH),'--raw-root',str(raw),'--output',str(out),'--tasks','humaneval'])
    with pytest.raises(ValueError,match='hash mismatch'):m.main()
    assert not out.exists()
