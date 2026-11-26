# AML – Project 2: Pose-Agnostic Anomaly Detection with Gaussian Splatting

Progetto del corso di Advanced Machine Learning (Politecnico di Torino, a.a. 2026-27).
Riproduciamo **SplatPose** sul dataset **MAD-Sim** e lo estendiamo con difetti sintetici creati modificando lo splat
e con un detector addestrato su di essi.

Il codice di partenza è il repository ufficiale di SplatPose ([m-kruse98/SplatPose](https://github.com/m-kruse98/SplatPose)),
il cui README originale è in [README_SPLATPOSE.md](README_SPLATPOSE.md). La licenza del modulo Gaussian Splatting è in
[LICENSE.md](LICENSE.md) (uso non commerciale / ricerca).

## Team e responsabilità

| Persona | Ruolo | Cartelle principali |
| --- | --- | --- |
| _nome_ | P1 – Setup e ricostruzione (Parte 1) | codice SplatPose, `notebooks/` |
| _nome_ | P2 – Pose estimation, valutazione (Parte 1) | codice SplatPose, `results/` |
| _nome_ | P3 – Editing dello splat (Parte 2) | `splat_editing/` |
| _nome_ | P4 – Detector single/two-stream (Parte 3) | `detector/` |
| _da decidere_ | Estensione | `extensions/` |

## Struttura del repository

```
.                     codice originale di SplatPose (train_and_render.py, pose_estimation.py, gaussian_splatting/, ...)
submodules/           rasterizer CUDA e simple-knn (submodule git)
notebooks/            notebook Colab per lanciare gli esperimenti
splat_editing/        Parte 2: operazioni stain / missing / burr e generazione del dataset sintetico
detector/             Parte 3: head single-stream e two-stream, training e valutazione
extensions/           estensione scelta
results/              SOLO file piccoli: metrics.csv e tabelle per il report
```

## Cosa va su GitHub e cosa su Drive

| GitHub (questo repo) | Google Drive (cartella `aml-project-2`) |
| --- | --- |
| Codice `.py`, notebook **senza output** | Dataset MAD-Sim / MAD-Real (`.zip`) |
| `results/metrics.csv`, piccole tabelle | Splat addestrati (`.ply`), checkpoint (`.pth`) |
| README e documentazione | Pose stimate, reference, anomaly map, dataset sintetico |

Il dataset MAD non va mai pubblicato qui (repo pubblico + diritti del dataset). Il `.gitignore` blocca già i
formati pesanti più comuni, ma controllate sempre `git status` prima di un commit.

## Primo avvio

```bash
git clone --recursive https://github.com/<owner>/aml-project-2.git
cd aml-project-2
```

`--recursive` scarica anche i submodule CUDA. Se avete già clonato senza: `git submodule update --init --recursive`.

## Come lavoriamo con git

1. **Mai lavorare direttamente su `main`.** Ogni attività ha il suo branch, con il prefisso del ruolo:
   `p1/gs-training`, `p2/eval-metrics`, `p3/stain-editing`, `p4/two-stream-head`.
2. Prima di iniziare: `git switch main && git pull`, poi `git switch -c p3/nome-attivita`.
3. Commit piccoli e frequenti, messaggi chiari: `Add stain editing on SH DC colors`.
4. Push del branch (`git push -u origin p3/nome-attivita`) e apertura di una **Pull Request** verso `main`.
5. Almeno **un altro membro** rilegge la PR prima del merge. Chi ha aperto la PR fa il merge.
6. Dopo il merge: cancellare il branch e ripartire dal punto 2.

**Notebook:** prima del commit fate *Modifica → Cancella tutti gli output* (in Colab) o *Clear All Outputs*.
Gli output rendono i diff illeggibili, gonfiano il repo e causano conflitti.

**Da Colab:** il modo più semplice è modificare il codice in locale e usare Colab solo per eseguire
(`git clone` / `git pull` del branch). Per salvare un notebook direttamente da Colab:
*File → Salva una copia in GitHub*, scegliendo il proprio branch (mai `main`).

**Interfacce condivise** (formato pose, reference, maschere, funzione di valutazione): ogni modifica passa da una
PR rivista da chi ne dipende.
