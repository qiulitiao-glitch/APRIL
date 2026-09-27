"""Compare paper claims, snapshot/views, frozen evidence and public defaults.

Exit 2 means incomplete or inconsistent. It is never converted to a pass merely
because the known-value checks passed. No GPU or restricted source is needed.
"""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import yaml

ROOT=Path(__file__).resolve().parents[1]
U='HYPERPARAMETER_VALUE_UNRESOLVED'


def load(path):return json.loads(path.read_text(encoding='utf-8'))


def leaves(value,path=''):
    if isinstance(value,dict):
        for k,v in value.items():yield from leaves(v,f'{path}.{k}' if path else k)
    elif isinstance(value,list):
        for i,v in enumerate(value):yield from leaves(v,f'{path}[{i}]')
    else:yield path,value


def audit():
    master=load(ROOT/'configs/paper/PAPER_FROZEN_CONFIG.json')
    sections=master['sections'];main=sections['april_main']
    checks=[]
    def check(name,observed,expected,source):
        checks.append(dict(check=name,status='PASS' if observed==expected else 'DISAGREEMENT',
                           observed=observed,expected=expected,source=source))
    for name,body in sections.items():
        if name=='statistics':continue
        view=yaml.safe_load((ROOT/f'configs/paper/{name}.yaml').read_text(encoding='utf-8'))
        check(f'{name} YAML projection',view['settings'],body,'PAPER_FROZEN_CONFIG.json')
    for name,source in master['sources'].items():
        p=ROOT/source['public_evidence']
        check(f'{name} public evidence digest',hashlib.sha256(p.read_bytes()).hexdigest(),
              source['public_evidence_sha256'],'recorded evidence identity')
    evidence=lambda name:load(ROOT/master['sources'][name]['public_evidence'])
    check('all frozen controller fields',main['controller_parameters'],evidence('controller'),'controller.json frozen evidence')
    check('all frozen factory fields',main['generation_factory_all_fields'],evidence('factory'),'factory.json frozen evidence')
    check('legacy controller view',load(ROOT/'configs/controller.json'),main['controller_parameters'],'authoritative snapshot')
    for key in ['dtype','attention_backend','cache_backend','greedy','batch_size','max_new_tokens','eos_enabled','seed']:
        check(f'main manifest {key}',main['generation'][key],evidence('main_manifest')[key],'historical main run manifest')
    paper=load(ROOT/'configs/paper/evidence/paper_claims.json')['claims']
    check('paper main E',main['E'],paper['E'],'paper section 5.1')
    check('paper main K',main['K'],paper['main_K'],'paper section 5.1')
    check('paper action order',main['A'],[[e,k] for k in main['K'] for e in main['E']],'frozen action pool')
    for gpu in ['a800','h100']:
        profile=sections['april_cross_gpu'][gpu]
        check(f'{gpu} paper K',profile['K'],paper['cross_K'],'paper section 5.1')
        check(f'{gpu} frozen generation fields',profile['generation_factory_all_fields'],
              evidence(gpu)['april_generation_values'],'historical cross-device config')
    for key in ['E','K']:
        check(f'paper LayerSkip {key}',sections['layerskip'][key],paper[f'layerskip_{key}'],'paper section 5.1')
    check('paper max_new_tokens',main['generation']['max_new_tokens'],paper['max_new_tokens'],'paper section 5.1')
    check('paper bootstrap replicates',sections['statistics']['bootstrap_replicates'],paper['bootstrap_replicates'],'paper section 5.1')
    sys.path.insert(0,str(ROOT))
    from april.config import GenerationConfig, EXIT_LAYERS, DRAFT_LENGTHS
    config=GenerationConfig()
    for field,expected in [('max_new_tokens',main['generation']['max_new_tokens']),
       ('margin_threshold',main['verification']['margin']),
       ('draft_confidence_threshold',main['shallow_stopping']['tr_nes_confidence_threshold']),
       ('draft_confidence_stop_after',main['shallow_stopping']['tr_nes_confidence_stop_after']),
       ('min_ngram',main['lookup']['min_ngram']),('max_ngram',main['lookup']['max_ngram']),
       ('max_candidates',main['lookup']['max_candidates'])]:
        check('public default '+field,getattr(config,field),expected,'snapshot-linked dataclass')
    check('runtime E',list(EXIT_LAYERS),main['E'],'snapshot')
    check('runtime K',list(DRAFT_LENGTHS),main['K'],'snapshot')
    tree=ast.parse((ROOT/'april/cli.py').read_text(encoding='utf-8'))
    defaults={}
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='add_argument':
            if node.args and isinstance(node.args[0],ast.Constant):
                for kw in node.keywords:
                    if kw.arg=='default':defaults[node.args[0].value]=ast.unparse(kw.value)
    check('CLI cap uses authoritative value',defaults.get('--max-new-tokens'),
          "PAPER['generation']['max_new_tokens']",'AST of public CLI')
    for name in ['run_april','run_ar','run_layerskip','run_pld']:
        wrapper=(ROOT/'scripts'/(name+'.py')).read_text(encoding='utf-8')
        check(f'{name} uses shared CLI','from april.cli import main' in wrapper,True,'public command source')
    from regenerate_table2 import regenerate
    measured={(r['task'],r['method']):r for r in regenerate()}
    printed=load(ROOT/'configs/paper/evidence/printed_table2.json')
    for row in printed['rows']:
        for method,tps in row['tps'].items():
            got=measured[row['task'],method]
            check(f'printed table {row["task"]}/{method}',
                  [got['requests'],round(got['tps'],2)],[row['requests'],tps],
                  'Paper printed values versus hash-pinned per-request measurements; numeric agreement only')
    unresolved=[p for p,v in leaves(sections) if v==U]
    classifications={r['field']:r for r in master['unresolved_classification']}
    for field in unresolved:
        check('classified unknown '+field,field in classifications,True,'explicit impact classification')
    for field in ['confidence_metric','threshold_adjustment_rule','exact_stopping_condition','eos_condition']:
        check('recovered '+field,main['shallow_stopping'][field]!=U,True,'frozen runtime behavior facts')
    check('prefill timed',main['generation']['timing']['prefill_included'],True,'frozen caller')
    supplemental=evidence('budget_statistics')
    check('supplemental statistics',sections['statistics']['supplemental'],supplemental,'recovered analysis source and archived bootstrap records')
    for gpu in ['a800','h100']:
        check(gpu+' statistical projection',sections['april_cross_gpu'][gpu]['statistics'],supplemental[gpu],'supplemental statistics evidence')
    blocking=[r for r in classifications.values() if r['category']=='BLOCKING_RUNTIME']
    passed=all(c['status']=='PASS' for c in checks) and not blocking
    return {'status':'PASS' if passed else 'FAIL','checks':checks,
            'unresolved_parameters':[classifications[x] for x in unresolved],
            'scope':'Runtime/config consistency. Historical provenance unknowns and required paper wording corrections remain explicit; this does not certify GPU equivalence.'}

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    sys.path.insert(0,str(ROOT))
    from april.config import require_frozen_config
    require_frozen_config(a.config)
    result=audit()
    with a.output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps({'status':result['status'],'checks':len(result['checks']),'disagreements':sum(c['status']!='PASS' for c in result['checks']),'unknown_fields':len(result['unresolved_parameters'])}))
    return 0 if result['status']=='PASS' else 2

if __name__=='__main__':raise SystemExit(main())
