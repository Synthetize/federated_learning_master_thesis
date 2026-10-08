# Tesi magistrale — stress test di Federated Learning

Contesto per una sessione nuova in questo repo. Chi legge non ha memoria delle chat
precedenti: qui c'e' il minimo per non rompere nulla. Branch di lavoro: **v2**.

## Di cosa si tratta

Tesi: *"Resilience and Critical Failure Points of Federated Learning: A Stress-Test on
Data Heterogeneity and Privacy Noise"* (Universita' di Camerino). Si misura quanto costa,
in **valore decisionale** e non in accuratezza, mettere insieme eterogeneita' dei dati
(Dirichlet alpha) e rumore di privacy (DP-SGD record-level, epsilon).

- **RQ1** — quanto costa ciascuno dei due fattori e quanto costano insieme, decomposto in
  Federation / Heterogeneity / Privacy / Interaction, con l'identita' `U = F + H + P + I`
  esatta a ogni soglia.
- **RQ2** — in quali combinazioni (alpha, epsilon) il modello federato **smette di valere
  la pena**: Safe Zone a due condizioni (batte sia le strategie che non richiedono modello,
  sia la media dei modelli locali della stessa partizione) e Safe Range come corsa
  contigua piu' lunga di soglie sicure.

La metrica e' il **Net Benefit** della Decision Curve Analysis (Vickers & Elkin 2006),
mediato sulla finestra clinica `p_t in [0.01, 0.20]`. **Non e' l'accuratezza, e non va
sostituito con l'accuratezza**: tutta la tesi poggia su questa scelta. Il caso che la
giustifica: alpha=10 con epsilon=8 ha accuratezza 0.946, la piu' alta fra le configurazioni
con rumore, e Safe Range 0.00.

## Stack

FedProx (mu = 0.01) su Flower + Ray in simulazione, 6 client, 100 round. DP-SGD
record-level via Opacus con accountant RDP. Dataset: malaria cell images (27.558 crop in
`cell_images_32/`), binario, bilanciato. Rete `Net`: 3 conv + GroupNorm, 102.082 parametri,
input 32x32. GroupNorm e non BatchNorm perche' Opacus rifiuta BatchNorm.

## Struttura

| cosa | dove |
|---|---|
| sweep federato | `federated/` (package: config, paths, model, dataset, privacy, app, sweep, report) |
| baseline locali e centralizzati (uno per seme) | `baselines/` (config, paths, data, training, local, central, summary) |
| analisi DCA, figure, tabelle LaTeX | `dca.py` |
| il notebook sostituito, con le diagnostiche citate in tesi | `legacy/` |
| capitoli LaTeX | altro repo: `master_thesis_overleaf/tesi_unicam_template/chapters/` |

```bash
python -m federated.sweep --help
python -m federated.sweep --preflight-only
python -m federated.sweep
python -m baselines
python -m federated.report --plots
python dca.py
```

## Stato attuale

Il **Capitolo 5 della tesi e' chiuso** sui dati a 3 semi (8.630 parole, tre revisioni
passate, sezione 5.4 con 15 citazioni). Il riallenamento in corso serve a chiudere cinque
limiti che il capitolo dichiara e non puo' risolvere, **senza cambiare framing**: niente
va riscritto a livello di metodologia o di domande di ricerca, solo i numeri.

In corso: sweep a **10 semi (42-51) x 6 alpha x 6 configurazioni di privacy = 360 run**,
da zero su una macchina con RTX 3060 Ti. I risultati a 3 semi in CPU sono stati cancellati
il 7 ottobre e restano nel commit `24166df2`.

## Invarianti — rompere questi invalida la tesi

1. **Non cambiare gli iperparametri.** `max_grad_norm = 5.0`, `momentum = 0.0`,
   `learning_rate = 0.03`, `fedprox_mu = 0.01`, `batch_size = 128`,
   `min_partition_size = 500`, `num_rounds = 100`. Ognuno e' giustificato in
   `federated/config.py` con una diagnostica, e sono citati in tesi. Cambiarne uno rende
   le run non confrontabili con nulla di quanto e' scritto.
2. **Una sola convenzione per i nomi delle cartelle:** `f"alpha_{alpha:g}"`, quindi 10.0 ->
   `alpha_10` e 1.0 -> `alpha_1`. Vale per `federated/paths.py` e per `baselines/paths.py`
   (che riusa `alpha_tag`). Prima convivevano `alpha_10` e `alpha_10.0`, che `dca.py` normalizza
   entrambi con `float()` e che quindi collassavano sulla stessa chiave sovrascrivendosi.
3. **Tre famiglie per ogni seme.** Lo split 80/20 del test set dipende dal seme, quindi
   ogni seme nuovo vuole lo sweep FL **piu'** `python -m baselines` (locali e centralizzato).
   Se ne manca una, la DCA scarta quel seme.
4. **Mai un solo hardware a meta' campione.** Tutti i 10 semi vanno girati sulla stessa
   macchina: mescolare CPU e GPU dentro il campione introdurrebbe una differenza
   sistematica in quello che si sta stimando.
5. **Platt solo sul validation pooled**, mai sul test. Il validation pooled e'
   `federated.dataset.load_pooled_valset`: i sei split 80/20 locali concatenati, mai nel
   test centralizzato.

## Trappole che hanno gia' morso

- **Il centralizzato e' uno per seme, mai uno solo.** Il vecchio `run_central()` allenava
  un solo modello sul seme 42, ma il test set dipende dal seme: valutato sul seme 43 dava
  accuracy da caso. Il codice e' stato rimosso (resta nella storia, commit `24166df2`).
- **I baseline non hanno piu' una copia della configurazione.** `baselines/config.py`
  legge tutto da `federated.config`, e `baselines/data.py` riusa `federated.dataset`.
  Non reintrodurre costanti ricopiate a mano: e' il disallineamento che il refactor
  elimina. Il refactor e' stato verificato bit a bit contro il vecchio `baselines.py`.
- **Le medie di conteggi non si riportano come conteggi.** Un "3.4 client su 6" era una
  media su 20 soglie x 3 draw. Se un numero e' un conteggio, va riportato intero o come
  frazione dichiarata di confronti.
- **`set_seeds` non risemina gli attori Ray ne' Opacus**, quindi le run non sono
  bit-riproducibili. Non e' un bug da sistemare: e' quello che rende gratis le ripetizioni
  a partizione fissa (`--replicates`), ed e' dichiarato fra i limiti.
- Le probabilita' di test sono salvate **round per round** in `probs_eps<eps>.f32`
  (float32 accodati, `np.fromfile(...).reshape(-1, n_test)`). Per scegliere un round non
  serve nessun checkpoint.
- **Fine riga:** il repo aveva i file con CRLF sul disco e LF in git, quindi ogni modifica
  appariva come file intero riscritto. Normalizzato a LF il 7 ottobre, con
  `.gitattributes` (`* text=auto eol=lf`). Non reintrodurre CRLF.

## Cosa resta da fare dopo lo sweep

1. Script di ricalibrazione: Platt stimato sul validation pooled, applicato alle
   probabilita' di test, DCA rieseguita. **Non richiede riallenamento.**
2. `dca.py`: da **half-range** a **media +- 1.96*SE**, e via il readability margin da 0.02
   costruito a mano. Con 10 semi diventano inutili.
3. Simmetria dell'early stopping: `argmax(val_accuracy)` dallo stream del validation
   (colonna in `training_history.csv`), e la riga corrispondente dallo stream del test.
4. Ripetizioni a partizione fissa: `python -m federated.sweep --replicates 4 --alphas 0.3
   --seeds 42 --epsilons 0.5 8`.
5. Riscrittura dei numeri nel Capitolo 5.

## Contesto piu' ampio

Il progetto Claude "Master Thesis" ha sei documenti di handoff e 41 paper, ma **Claude Code
non vede i progetti claude.ai**. Se serve il contesto completo, va copiato qui a mano.
