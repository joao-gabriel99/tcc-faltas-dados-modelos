"""Common evaluation: metrics, confusion matrix, CV."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    LeaveOneGroupOut,
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)


def metrics(y_true, y_pred) -> dict:
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
    }


def cv_evaluate(
    model_factory,
    X,
    y,
    n_splits=5,
    seed=42,
    groups=None,
    group_mode: str | None = None,
) -> dict:
    """Avalia por estratos ou por grupos inteiros para impedir vazamento."""
    if group_mode == 'leave_one_group_out':
        if groups is None:
            raise ValueError('groups é obrigatório no modo leave_one_group_out')
        splits = LeaveOneGroupOut().split(X, y, groups)
    elif group_mode == 'stratified_group':
        if groups is None:
            raise ValueError('groups é obrigatório no modo stratified_group')
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=seed
        )
        splits = splitter.split(X, y, groups)
    elif group_mode is None:
        splitter = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=seed
        )
        splits = splitter.split(X, y)
    else:
        raise ValueError(f'Modo de agrupamento desconhecido: {group_mode}')

    scores = {k: [] for k in ['accuracy', 'f1_macro', 'f1_weighted', 'precision_macro', 'recall_macro']}
    held_out_groups = []
    for fold, (tr, te) in enumerate(splits):
        if groups is not None:
            overlap = set(np.asarray(groups)[tr]) & set(np.asarray(groups)[te])
            if overlap:
                raise RuntimeError(f'Vazamento de grupos no fold {fold}: {overlap}')
            held_out_groups.append(
                sorted({str(value) for value in np.asarray(groups)[te]})
            )
        m = model_factory()
        m.fit(X[tr], y[tr])
        p = m.predict(X[te])
        for k, v in metrics(y[te], p).items():
            scores[k].append(v)
    result = {
        k: (float(np.mean(v)), float(np.std(v))) for k, v in scores.items()
    }
    result['fold_count'] = len(scores['accuracy'])
    if groups is not None:
        result['held_out_groups'] = held_out_groups
    return result


def fit_and_report(
    model,
    X,
    y,
    test_size=0.2,
    seed=42,
    groups=None,
    test_group=None,
) -> dict:
    held_out_group = None
    if groups is None:
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=seed
        )
    else:
        groups_array = np.asarray(groups)
        if test_group is None:
            raise ValueError('test_group é obrigatório quando groups é informado')
        held_out_group = test_group
        test_mask = groups_array == held_out_group
        if not np.any(test_mask) or np.all(test_mask):
            raise ValueError(f'Grupo de teste inválido: {held_out_group}')
        Xtr, Xte = X[~test_mask], X[test_mask]
        ytr, yte = y[~test_mask], y[test_mask]
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    return {
        'metrics': metrics(yte, pred),
        'confusion_matrix': confusion_matrix(yte, pred).tolist(),
        'classes': sorted(set(yte)),
        'report': classification_report(yte, pred, zero_division=0, output_dict=True),
        'y_test': yte,
        'y_pred': pred,
        'model': model,
        'held_out_group': None if held_out_group is None else str(held_out_group),
    }
