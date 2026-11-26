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
results/              SOLO le tabelle finali per il report (il metrics.csv "vivo" sta su Drive)
```

## Cosa va su GitHub e cosa su Drive

| GitHub (questo repo) | Google Drive (cartella `aml-project-2`) |
| --- | --- |
| Codice `.py`, notebook **senza output** | Dataset MAD-Sim / MAD-Real (`.zip`) |
| Tabelle finali per il report (`results/`) | Splat addestrati (`.ply`), checkpoint (`.pth`), `metrics.csv` |
| README e documentazione | Pose stimate, reference, anomaly map, dataset sintetico |

Il dataset MAD non va mai pubblicato qui (repo pubblico + diritti del dataset). Il `.gitignore` blocca già i
formati pesanti più comuni, ma controllate sempre `git status` prima di un commit.

## Come lavoriamo su Google Drive

La cartella condivisa è `aml-project-2`. Ognuno la aggiunge al proprio Drive (*Condivisi con me* → clic destro →
*Organizza → Aggiungi scorciatoia → Il mio Drive*), così su Colab il percorso è uguale per tutti:
`/content/drive/MyDrive/aml-project-2`.

Principio: **ognuno scrive solo nella propria zona e legge da quelle degli altri; non si sovrascrive mai nulla**
(una nuova versione va in una nuova cartella).

| Cartella | Contenuto | Chi scrive | Chi legge |
| --- | --- | --- | --- |
| `data/` | `MAD-Sim.zip` originale, `MAD-Sim_3dgs.zip` (da `prepare_lego_ad.py`), `model.zip` (retrieval per `PAD_utils`) | P1, una volta sola | Tutti, a ogni sessione |
| `runs/` | Output di SplatPose (Parte 1): splat `.ply`, pose stimate, reference, anomaly map, metriche per classe | P1, P2 | P3 (splat), P4 (pose e reference) |
| `synthetic/` | Dataset sintetico (Parte 2): immagini con difetto, reference, maschere, pose, metadati | P3 | P4 |
| `heads/` | Checkpoint del detector (Parte 3) con la configurazione usata | P4 | Tutti |
| `results/` | `metrics.csv` (una riga per esperimento) e figure qualitative per il report | Tutti, solo aggiungendo righe | Tutti |

`data/` è **in sola lettura**: dopo il primo caricamento non si modifica più.

### Nomi delle cartelle

Il nome dice con quali impostazioni è stato prodotto il contenuto:

```
runs/repro_v210_it30k_k175/02Unicorn/        riproduzione con impostazioni di default
runs/abl_views100_it30k_k175/02Unicorn/      ablation: 100 viste di training
synthetic/v1_stain-missing-burr/02Unicorn/   prima versione del dataset sintetico
heads/two-stream_syn-v1_effnet/              head two-stream addestrato su synthetic v1
```

- In ogni cartella di esperimento c'è un `info.json` con autore, data, impostazioni e **hash del commit git** usato.
- Concordiamo una cartella di riferimento per la Parte 1 (es. `runs/repro_v210_it30k_k175/`): P3 e P4 lavorano su
  quella e da quel momento nessuno la modifica.
- Le estensioni usano lo stesso schema con prefisso `ext_` (es. `runs/ext_dinov2_.../`).

### Regole pratiche con Colab

1. **I dati si copiano, non si leggono da Drive**: a inizio sessione copiare gli zip da `data/` in `/content` e
   decomprimerli lì. Leggere migliaia di immagini direttamente da Drive è lentissimo.
2. **Salvare su Drive appena un passo finisce** (splat, pose, reference, metriche): se Colab si disconnette si perde
   solo il passo in corso.
3. **Tante immagini piccole → un `.zip` per classe** (dataset sintetico, reference): Drive le gestisce molto meglio.
4. **Il codice non sta su Drive**, sta su GitHub. Su Drive solo dati e risultati.
5. **Non cancellare il lavoro degli altri**: se qualcosa è sbagliato, scriverlo nel gruppo. Il cestino di Drive tiene i
   file 30 giorni.
6. **Un solo `metrics.csv`**: quello vivo è su Drive in `results/`. Alla fine si copiano in `results/` di questo repo
   solo le tabelle finali del report.

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
