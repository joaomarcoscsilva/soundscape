import pickle
import argparse
from jax import numpy as jnp
import jax

jax.numpy.zeros(10)

from calibrate import transform
from metrics import reorder_id, balacc

from soundscape import dataset, settings

with settings.Settings.from_file("settings/supervised.yaml"):
    ws = dataset.get_class_weights()
    vds = dataset.get_dataset_dict('val', jax.random.PRNGKey(0))
    vls = vds['labels'][vds['id'].argsort()]

    tds = dataset.get_dataset_dict('test', jax.random.PRNGKey(0))
    tls = tds['labels'][tds['id'].argsort()]

def ensemble(logits, ens_type, kept_idx = None):
    if kept_idx is not None:
        logits = logits[kept_idx]

    if ens_type == 'logits':
        return logits.mean(0)

    elif ens_type == 'probs':
        return jnp.log(jax.nn.softmax(logits).mean(0))
    
    elif ens_type == 'votes':
        logits = jnp.log(jax.nn.softmax(logits) * ws)
        return jnp.log(jax.nn.softmax(logits * 1000).mean(0))

    else:
        raise ValueError(f'Unknown ens_type {ens_type}')        


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("logs")
    parser.add_argument("preds")
    parser.add_argument('--splitminute', action='store_true', default=False)
    parser.add_argument('--nosave', action='store_true', default=False)

    args = parser.parse_args()

    with open(args.logs, "rb") as f:    
        logs = pickle.load(f)

    with open(args.preds, "rb") as f:
        preds = pickle.load(f)

    with open(args.preds.replace('val', 'test'), 'rb') as f:
        test_preds = pickle.load(f)

    with open(args.preds.replace('val', 'u'), 'rb') as f:
        u_preds = pickle.load(f)

    accs = logs['selected_nb']['balacc']['val']

    logits = preds['logits']
    ids = preds['id']

    test_logits = test_preds['logits']
    test_ids = test_preds['id']

    u_logits = u_preds['logits']
    u_ids = u_preds['id']

    # u_logits = u_logits.reshape((u_logits.shape[0], u_logits.shape[1] // 20, 20, -1))

    logits = reorder_id(logits[None, ...], ids[None, ...])[0]
    ids = reorder_id(ids[None, ...], ids[None, ...])[0]

    test_logits = reorder_id(test_logits[None, ...], test_ids[None, ...])[0]
    test_ids = reorder_id(test_ids[None, ...], test_ids[None, ...])[0]

    u_logits = reorder_id(u_logits[None, ...], u_ids[None, ...])[0]
    u_ids = reorder_id(u_ids[None, ...], u_ids[None, ...])[0]

    best_of_all = 0
    best_keys = {}

    for split in ['', 'notemp_bias', 'scalar_bias', 'vector_bias', 'scalar_nobias', 'vector_nobias']:
        if split != '':
            params = logs['calibration'][split]
            sel_epochs = logs['selected_epochs_nb']['val']
            params = {k:v[jnp.arange(v.shape[0]), sel_epochs] for k,v in params.items()}
            cal_logits = jax.vmap(transform)(params, logits)
        else:
            cal_logits = logits

        for i in range(32):
            ens_logits = cal_logits

            ens_logits = ens_logits[accs.argsort()]
            ens_logits = ens_logits[i:]

            avglogits = ens_logits.mean(0)
            problogits = jnp.log(jax.nn.softmax(ens_logits).mean(0))
            votelogits = jnp.log(jax.nn.softmax(ens_logits*1000).mean(0))

            avgacc = balacc(avglogits, vls)
            probacc = balacc(problogits, vls)
            voteacc = balacc(votelogits, vls)

            if avgacc > best_of_all:
                best_of_all = avgacc
                best_keys = {'ens_type': 'logits', 'split': split, 'idx': i}
            
            if probacc > best_of_all :
                best_of_all = probacc
                best_keys = {'ens_type': 'probs', 'split': split, 'idx': i}

            if voteacc > best_of_all :
                best_of_all = voteacc
                best_keys = {'ens_type': 'votes', 'split': split, 'idx': i}

    print(best_of_all)
    print(best_keys)

    split = best_keys['split']
    i = best_keys['idx']
    ens_type = best_keys['ens_type']

    if split != '':
        params = logs['calibration'][split]
        sel_epochs = logs['selected_epochs_nb']['val']
        params = {k:v[jnp.arange(v.shape[0]), sel_epochs] for k,v in params.items()}
        cal_test_logits = jax.vmap(transform)(params, test_logits)
        cal_u_logits = jax.vmap(transform)(params, u_logits)
    else:
        cal_test_logits = test_logits
        cal_u_logits = u_logits

    cal_test_logits = cal_test_logits[accs.argsort()]
    cal_test_logits = cal_test_logits[i:]

    cal_u_logits = cal_u_logits[accs.argsort()]
    cal_u_logits = cal_u_logits[i:]

    avglogits = cal_test_logits.mean(0)
    problogits = jnp.log(jax.nn.softmax(cal_test_logits).mean(0))
    votelogits = jnp.log(jax.nn.softmax(cal_test_logits*1000).mean(0))

    avgulogits = cal_u_logits.mean(0)
    probulogits = jnp.log(jax.nn.softmax(cal_u_logits).mean(0))
    voteulogits = jnp.log(jax.nn.softmax(cal_u_logits*1000).mean(0))

    avgacc = balacc(avglogits, tls)
    probacc = balacc(problogits, tls)
    voteacc = balacc(votelogits, tls)


    if ens_type == 'logits':
        print(avgacc)
        saved_logits = avgulogits

    if ens_type == 'probs':
        print(probacc)
        saved_logits = probulogits

    if ens_type == 'votes':
        print(voteacc)
        saved_logits = voteulogits

    saved_dict = {
        'id': u_ids,
        'one_hot_labels': jax.nn.softmax(saved_logits)
    }

    if not args.nosave:
            
        with open(f"data/resnetb_ens_labels.pkl", "wb") as f:
            pickle.dump(saved_dict, f)
