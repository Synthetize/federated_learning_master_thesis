"""Stress test federato: eterogeneita' dei dati (Dirichlet alpha) x rumore di privacy
(DP-SGD record-level, epsilon), su classificazione di cellule di malaria.

Sostituisce il notebook `malaria_dpsgd.ipynb` (conservato in `legacy/` perche' contiene le
diagnostiche D1-D5 e il test sull'ottimizzatore, citati in tesi).

    python -m federated.sweep --help
    python -m federated.report

Moduli, in ordine di dipendenza:

    config     ExperimentConfig, CONFIG, set_seeds, guardia sull'accounting DP
    paths      tutti i percorsi di output, convenzione dei nomi, indice replicate
    model      Net, train (DP-SGD + termine prossimale), test, evaluate_with_probs
    dataset    dataset, partizionamento Dirichlet, loader, preflight
    privacy    sigma per client, anteprima rumore/segnale, tabella sigma
    app        ClientApp e valutazione centralizzata di Flower
    sweep      una run, salvataggio, ciclo dello sweep, CLI
    report     tabelle aggregate e figure diagnostiche

Non si importa niente di pesante qui: `import federated` non deve tirare dentro torch,
flwr o opacus, cosi' leggere la configurazione resta istantaneo.
"""

__all__ = ["config", "paths", "model", "dataset", "privacy", "app", "sweep", "report"]
