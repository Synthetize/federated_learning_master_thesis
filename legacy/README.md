# legacy/

## `malaria_dpsgd.ipynb`

Il notebook con cui sono state prodotte le run a tre semi. Sostituito dal package
`federated/` (7 ottobre 2026), ma **conservato e non cancellato** perche' contiene
materiale citato in tesi che nel package non ha senso portare:

- **sezione 7.1** - diagnostiche D1-D5 su perche' `alpha <= 0.5` falliva anche senza DP;
- **sezione 7.2** - verifica che il termine prossimale di FedProx venga davvero applicato;
- **sezione 7.3** - diagnostica sull'ottimizzatore che fa divergere i client, da cui viene
  la scelta `momentum = 0.0`;
- gli **output salvati** di tutte le celle, cioe' l'evidenza dei numeri riportati nei
  commenti di `federated/config.py` (0.961 -> 0.952 a alpha=10, 0.500 -> 0.703 a
  alpha=0.5 seed 42, epsilon 8/16/32 che danno 0.922/0.925/0.930).

Le tre celle di diagnostica erano **gia' interamente commentate** nel notebook, quindi non
facevano parte dello sweep: per questo non sono state portate nel package.

Non eseguirlo. Per allenare si usa `python -m federated.sweep`.
