# comnumpy — Architecture & Decisions, v0.6

Statut : draft — 2026-08-09. **Remplace ADD v0.5.**
Périmètre : de l'état constaté sur `main` (18 commits) à la soumission JOSS.
Ce document est normatif : le code se conforme aux décisions D-x ou les
amende explicitement. Les décisions amendées conservent leur numéro et
portent la mention *Amendé v0.x* avec le motif.

Nouveautés v0.6 : décision **D43** (modèles de canaux normalisés,
§4.12) — profils de retard catalogués et processus Doppler, troisième
application du motif D15/D17/D20.

Nouveautés v0.5 : décision **D42** (observation et câblage de chaîne,
§4.11) — la chaîne ne décrit plus que le système de communication, les
blocs d'instrumentation disparaissent ; amendement de **D11** (les
*monitors* sont supprimés, non convertis en loggers) ; amendement de
**D22** (vocabulaire arrêté : `reference` et famille `DataAided*`) ;
mise à jour de la cible d'usage de **D40c** et du point ouvert « graphe
de calcul » de la §8, dont D42 traite le cas de la seconde entrée ;
**annexe A.5** — les défauts levés par la conversion des docstrings
au gabarit §4.10, qui oblige à relire chaque bloc ligne à ligne.

Nouveautés v0.4 : décisions **D34–D35** (reconfiguration des paramètres
et exécution de balayages, §4.8) ; **D36–D41** (ergonomie d'API et
qualité de code, §4.9) ; amendement de **D10** (docstring comme support
de cours, gabarit normatif en §4.10) ; mise à jour de la §7 ; points
ouverts « runner de simulation » et « surface d'API » ajoutés à la §8.

Nouveautés v0.3 : principe P7 ; §3 « état constaté » (audit exécuté du
2026-07-28) ; décisions **D15–D21** (allocation spectrale OFDM et WDM),
**D22–D24** (conventions d'estimateurs), **D25–D27** (rendu graphique),
**D28–D29** (structure de trame), **D30** (langue),
**D31–D33** (introspection et sérialisation) ;
amendements de D1, D2, D7, D8, D9, D11 ; annexe A (défauts constatés).

---

## 1. Principes

| # | Principe | Conséquence concrète |
|---|----------|----------------------|
| P1 | **Léger** — numpy + scipy + matplotlib, rien d'autre | *Amendé v0.3* : matplotlib est assumé comme dépendance dure (les figures font partie du produit pédagogique). Toute autre dépendance est un amendement de ce document |
| P2 | **Lisible avant rapide** — la lib est aussi un objet pédagogique | Un algorithme illisible mais 2× plus rapide perd ; les boucles temps irréductibles restent en Python clair |
| P3 | **Validé, visiblement** — la justesse est le produit | Tout algorithme est accompagné d'un golden test et, si pertinent, d'un script `validation/`. **Aucun exemple numérique n'entre dans une docstring sans avoir été exécuté** |
| P4 | **Reproductible** — graine → mêmes courbes | RNG local par bloc (`default_rng`), graine de chaîne dérivée (D6) |
| P5 | **Formes explicites, jamais devinées** — pas de promotion silencieuse | Un bloc qui exige un axe le vérifie et refuse ; le message d'erreur donne la forme attendue |
| P6 | **Une place pour chaque chose** — convention d'axes unique | CONVENTIONS.md ; deux layouts canoniques, trois catégories de blocs |
| P7 | **Compréhensible sans documentation** *(nouveau)* | Types nommés plutôt qu'entiers magiques ; `__repr__` qui montre l'objet plutôt que de le décrire ; grandeurs estimées distinguables des grandeurs configurées par leur seul nom |

---

## 2. Convention de tenseurs (normatif — détail dans CONVENTIONS.md)

**Layouts canoniques :**

- Série : `(..., N)` — échantillons/bits sur l'axe −1.
- Bloc : `(..., T, F)` — bloc en −2, contenu du bloc en −1.
- Conversion série ↔ bloc = `reshape` pur (C-order). Toute implémentation
  de S/P exigeant une transposition viole la convention.

**Axes structurels optionnels**, à gauche du cœur, ordre d'imbrication
physique : `(batch..., wdm, ant/pol, cœur)`. Le batch n'est jamais un axe
nommé : il est implicite dans `...` et transporté par broadcasting.

**Domaine (temps/fréquence) :** la FFT change la signification de l'axe −1,
jamais sa position. La sémantique courante est documentée dans la docstring
des blocs qui la changent.

**Ordre spectral :** les allocations et masques fréquentiels sont décrits
en **ordre physique** (indice signé, DC au centre). La conversion vers
l'ordre FFT est explicite, unique, et se fait par `ifftshift` (D16).

**Catégories de blocs** (chaque bloc en déclare une) :

1. *Élément* — opère point à point, ignore la forme (AWGN, mapper).
2. *Axe −1* — `axis=-1` en dur, broadcast sur le reste (filtres, FFT).
3. *Axe déclaré* — exige son axe (MIMO sur `ant`, mux sur `wdm`, S/P,
   allocation de porteuses), validation dans `prepare()`.

**L'attribut `is_mimo` est supprimé.** Il est le mécanisme actuel de
devinette de forme, c'est-à-dire exactement ce que P5 interdit. Il est
remplacé par la catégorie d'axes déclarée et la validation en
`prepare()`.

---

## 3. État constaté (audit exécuté du 2026-07-28)

Mesures reproductibles, sur `main` à 18 commits. Elles motivent les
amendements de la §4.

| Indicateur | Valeur |
|---|---|
| Lignes dans `src/` | ~7 500 |
| Tests unitaires | 20, tous verts, 5,5 s |
| Couverture par module | `core` 7 fichiers · `mimo` 1 · `ofdm` 1 · **`optical` 0** |
| Doctests | **48 échecs sur 218** |
| Version PyPI | **0.91** (0.8 → 0.9 → 0.91, dernière publication 2026-02-25) |
| Dépendances déclarées | numpy, scipy, matplotlib, seaborn, tqdm |
| — dont `tqdm` | **0 occurrence** dans `src/` |
| — dont `seaborn` | **1 occurrence** (`sns.kdeplot`, `core/visualizers.py:327`) |
| Occurrences de `is_mimo` | ~19 |
| Appels `print()` dans `src/` | 19 |
| Exemple du README | **ne s'exécute pas** (deux défauts distincts) |
| `examples/mimo/one_shot_mimo.py` | ~15 s par point de SNR (détecteur ML scalaire) |

Trois constats structurants :

1. **`optical/` est le différenciateur revendiqué et le module le moins
   testé** : 0 test, et il contient déjà un bloc entièrement mort
   (`PhaseNoise`). Le risque de la §8 n'est pas théorique.
2. **Les docstrings ne sont pas exécutées.** `compute_PAPR` documente
   `2.0` et calcule `1.4606` ; `compute_evm` documente `0.0506` et
   calcule `0.0365`. Les implémentations sont cohérentes — ce sont les
   valeurs des exemples qui sont inventées. Pour une lib dont le produit
   est la justesse, c'est le défaut le plus coûteux en crédibilité.
3. **Le numéro de version planifié en v0.2 (`v0.2.0`) est une
   régression** vis-à-vis du 0.91 déjà publié : `pip install -U` ne
   l'aurait jamais vue.

Le détail fichier par fichier est en annexe A.

---

## 4. Journal de décisions

### 4.1 Décisions fondatrices (v0.2, amendées le cas échéant)

| # | Décision | Motif | Alternatives rejetées | Statut |
|---|----------|-------|-----------------------|--------|
| D1 | Cœur **numpy + scipy + matplotlib**. `seaborn` et `tqdm` sont retirés. Ni torch ni backend GPU. Backend graphique unique : cf. D26 | P1 ; les figures font partie du produit pédagogique, mais `seaborn` tirait `pandas` pour un unique `kdeplot` et `tqdm` n'était pas utilisé | numpy+scipy strict avec matplotlib en extra (rejeté : les figures sont centrales) ; backend torch ; Array API (scipy.signal et RNG non couverts — réévaluer ~2028) | **Amendé v0.3** |
| D2 | Convention d'axes §2, appliquée en une fois (breaking change), **incluant la suppression de `is_mimo`** | 7 étoiles : le coût de casse est nul aujourd'hui, jamais plus. L'audit montre que `mimo/` est déjà conforme `(ant, N)` ; seul `ofdm/` casse (S/P en `order="F"` renvoyant `(N_sub, M)`, soit le transposé du layout Bloc) | Migration progressive ; axes nommés xarray (viole P1) | **Amendé v0.3** |
| D3 | Accélération GPU éventuelle via compatibilité CuPy (`_backend.py` interne regroupant les appels scipy) | CuPy réimplémente l'API scipy ; portabilité sans toucher l'API publique | Array API (cf. D1) ; numba obligatoire | Acté, non prioritaire |
| D4 | Module `fec/` : encodeur convolutif + Viterbi (soft/hard), vectorisation **batch**, boucle Python sur le temps seul | Trou fonctionnel le plus visible ; prototype validé (gain de codage conforme, 2,5 s / 2·10⁵ bits soft K=7) | Implémentation C/Cython (coût wheels + viole P1/P2) ; numba requis | **Acté** — intégration à partir de `fec_proposal.py` |
| D5 | LDPC min-sum en tableaux `(batch, n_edges)`, fichier séparé, après D4 | Même philosophie de vectorisation ; ne pas bloquer la sortie dessus | — | Planifié |
| D6 | Graine de chaîne : `Sequential.seed(s)` dérive déterministiquement les graines des blocs stochastiques (`SeedSequence.spawn`) | P4 ; « reproduisez la figure à l'identique » est un argument JOSS | Graines manuelles par bloc uniquement (état actuel) | Acté |
| D7 | Dossier `validation/` : scripts autonomes traçant simulation vs référence. **Priorité absolue à `optical/`** | P3 ; sert 4 fois : tests golden, vitrine, figures JOSS, réponse aux reviewers. L'audit confirme 0 test sur le module différenciateur | Étendre uniquement `tests/` (invisible pour un visiteur) | **Amendé v0.3** |
| D8 | **Doctest global exécuté en CI**, pas seulement le README | *Amendé v0.3* : l'exemple du README a deux défauts (`SymbolMapper(M=16)` n'existe pas — la signature est `SymbolMapper(alphabet)` ; et `chain(10000)` ne renvoie qu'une sortie, pas un couple `tx, rx`). Et 48 doctests échouent ailleurs. Limiter D8 au README laisserait le gros du problème en place | Correction du seul README | **Amendé v0.3 — Urgent** |
| D9 | Citabilité : tag **`v1.0.0`**, release notes, `CITATION.cff`, CHANGELOG | *Amendé v0.3* : PyPI est à 0.91, donc `v0.2.0` serait une régression PEP 440. Le breaking change D2, la clarification des dépendances D1 et la stabilisation de l'API justifient un 1.0.0 franc plutôt qu'un 0.92 qui masquerait la rupture | `v0.2.0` (régression) ; `0.92` (n'annonce pas la rupture) | **Amendé v0.3** |
| D10 | Docstrings : numpydoc + section **Signal Model** en LaTeX obligatoire, catégorie d'axes déclarée, doctest court **et exécuté**. *Amendé v0.4* : la docstring est promue **support de cours** — gabarit normatif complet en §4.10, bijection symbole mathématique ↔ paramètre, table de notation unique dans CONVENTIONS.md, validation `numpydoc` en CI | P2/P3 ; uniformiser ce que les meilleurs fichiers font déjà. L'amendement v0.4 assume la double vocation pédagogie/recherche : la doc de référence doit se lire comme un polycopié — l'équation, puis chaque symbole retrouvé dans les paramètres | Doc de cours séparée du code (désynchronisation garantie, cf. les 48 doctests) | **Amendé v0.4** |
| D11 | `debug` par `logging`, plus de `print` dans `src/` | *Amendé v0.3* : le périmètre dépasse les blocs. `core/monitors.py` affiche par conception (11 `print`), `generics.py` et `mimo/channels.py` aussi. Les *monitors* deviennent des `logging.Logger` configurables, pas des `print`. *Amendé v0.5* : D42 les **supprime** plutôt que de les convertir — un bloc qui journalise reste un parasite dans `module_list`. Ce qui subsiste de D11 est intact (aucun `print` dans `src/`, débogage par `logging`) ; la mesure passe par `signal_report()`, qui rend un dictionnaire et laisse l'appelant choisir la présentation | — | **Amendé v0.5** |
| D12 | Demapper : sortie LLR (`soft=True`) en plus des décisions dures | Interface nécessaire à D4 (Viterbi soft) ; standard du domaine | Module LLR séparé | Acté |
| D13 | Extension différentiable : hors lib. Blocs torch natifs écrits à la main dans un dépôt séparé si besoin | La différentiabilité se conçoit (soft-demapping, pas d'argmin), ne se convertit pas ; P1 | torch dans le cœur ; conversion mécanique numpy→torch | Acté |
| D14 | Cible : soumission **JOSS**. Positionnement = léger + libre + module optique (SSFM/DBP), pas « framework général » | Créneau réel : ni Sionna ni MATLAB ne couvrent la fibre non linéaire en Python léger | Positionnement « alternative à Sionna » (perdant : leur Sequential est différentiable) | Acté |

### 4.2 Allocation spectrale (nouveau)

| # | Décision | Motif | Alternatives rejetées | Statut |
|---|----------|-------|-----------------------|--------|
| D15 | Une allocation est un objet `CarrierAllocation` (dataclass **frozen**) portant un masque `(T_p, N_fft)` de période `T_p`, plus ses métadonnées (espacement, CP, standard, clause de référence) | Les pilotes dispersés (DVB-T, CRS LTE, DM-RS NR) sont des motifs **2D temps-fréquence**, inexprimables par un masque 1D. Un tableau nu perd la provenance, et la provenance est ce qu'un reviewer vérifie | Tableau d'entiers nu (état actuel) ; dictionnaire de configuration non typé | **Acté** |
| D16 | Masque en **ordre physique** (indice signé, DC au centre) ; conversion vers l'ordre FFT explicite par `ifftshift`. Le paramètre `shift` est supprimé | P5/P6 : les standards s'écrivent en signé. Le `shift` actuel applique `fftshift` quand il vaut `False` — le nom dit l'inverse de l'effet. `ifftshift` est l'inverse correct (identique pour `N` pair, divergent pour `N` impair) | Garder `shift` avec sémantique corrigée (nom déjà brûlé) | **Acté** |
| D17 | Catalogue derrière une fonction unique `get_allocation(standard, **kwargs)`, adossée à un registre extensible par décorateur | Simple côté appelant ; un utilisateur ajoute son standard sans patcher la lib | Une classe par standard ; `if/elif` en dur | **Acté** |
| D18 | `CarrierAllocator` / `CarrierExtractor` sont de catégorie **Axe déclaré** : ils exigent le layout Bloc `(..., T, F)` et le valident dans `prepare()`. Les valeurs de pilotes sont un argument séparé du masque. Le paramètre `axis` disparaît | D2 ; le masque décrit *où*, les pilotes disent *quoi* — deux durées de vie différentes. Émetteur et récepteur partagent le même objet, ce qui supprime la classe de bugs « masques divergents » | Pilotes stockés dans l'allocation ; `axis` libre paramétrable | **Acté** |
| D19 | Grille WDM ITU-T G.694.1 : un canal est décrit par le couple d'entiers `(n, m)` — centre `193,1 + n × 0,00625` THz, largeur de slot `12,5 × m` GHz. **L'axe `wdm` est réservé au multiplexage/démultiplexage et interdit en entrée de tout bloc non linéaire** | Convention normative du domaine ; la grille flexible représente toutes les grilles fixes comme cas particuliers. Le second volet est un garde-fou physique : cf. §8 | Espacements en Hz passés à la main (non citable, non vérifiable) ; NLSE couplées par canal (autre modèle, non implémenté) | **Acté** |
| D20 | Chaque entrée du catalogue porte son `expect={"data": …, "pilots": …}` recopié du tableau du standard, vérifié à la construction, plus la clause en `reference`. Aucune entrée n'est fusionnée sans ces deux champs | P3 ; l'assertion vit à trois lignes de la valeur produite, donc l'entrée reste lisible **et** vérifiée d'un coup d'œil. Un test séparé qui répète les mêmes chiffres ailleurs se désynchronise | Fichier de tests golden séparé | **Acté** |
| D21 | Une allocation doit être lisible sans documentation : (a) `CarrierType` IntEnum, jamais d'entiers magiques ; (b) `__repr__` traçant une **carte spectrale ASCII** ; (c) constructeurs calqués sur les tableaux du standard | P7. Un ingénieur comprend une allocation en la **voyant**. Une carte ASCII fonctionne dans un terminal, un log, un message d'erreur et un doctest — là où une figure matplotlib ne fonctionne pas | Documentation prose seule ; `plot()` seul (inutilisable en CI et en débogage) | **Acté** |

### 4.3 Estimateurs (nouveau)

| # | Décision | Motif | Alternatives rejetées | Statut |
|---|----------|-------|-----------------------|--------|
| D22 | **Cycle de vie inspiré de scikit-learn.** Signature unique `fit(X, y=None)` où `y` est la référence connue (préambule, pilotes) ; `y=None` = estimation aveugle. `fit` retourne `self`. `partial_fit(X)` pour les algorithmes adaptatifs (CMA, RDE, décision-dirigé). `forward()` reste l'interface unique du `Processor`. *Amendé v0.5* : **vocabulaire arrêté** — le signal connu contre lequel un estimateur se cale s'appelle `reference` partout (paramètre du bloc, `sweep(reference=…)`, `wiring`), et les blocs qui en demandent un sont `DataAided*`, par opposition à la famille `Blind*` déjà nommée ainsi. Les classes `TrainedBased*` sont renommées en conséquence | La convention sklearn est la plus répandue en Python scientifique, et le couple supervisé/non-supervisé se superpose exactement à *data-aided*/*blind*. L'audit montre que `fit()` existe déjà sur 8 classes, mais que le second argument y désigne tour à tour `x_target`, `x_preamble` et `w0` (une initialisation d'algorithme, pas une donnée). L'amendement v0.5 clôt le même problème côté *nom* : `target_data` était un emprunt à scikit-learn, où la cible est ce qu'on **prédit** — ici c'est le signal **connu**. `reference` est le terme du domaine (3GPP *reference signal*), et surtout celui que `sweep()` employait déjà pour l'exacte même chose. Note : les *métriques* gardent `X_target` / `X_detected`, qui est la paire standard d'une mesure d'erreur (`y_true`/`y_pred` chez sklearn) — rôle différent, mot différent, les deux justes | Renommer `forward` en `transform` (fragmenterait `Sequential`) ; garder l'état actuel | **Amendé v0.5** |
| D23 | **Underscore final pour toute grandeur estimée à partir de données** : `theta_`, `h_`, `gain_`, `H_`, `n_iter_`. Les quantités dérivées des seuls paramètres n'en portent pas. `NotFittedError` si `forward` précède `fit` | P7 ; l'ambiguïté existe déjà et mord : le canal *estimé* `self.H` du compensateur MIMO est homonyme du canal *configuré* `H` de `FlatMIMOChannel`. Aujourd'hui un `forward` avant `fit` applique silencieusement `theta = 0` | Préfixe `est_` ; documentation seule | **Acté** |
| D24 | **Frontière explicite avec scikit-learn** : on emprunte les noms de méthodes et le cycle de vie, **jamais la disposition des données**. Le layout `(n_samples, n_features)` est incompatible avec D2 et n'entre pas dans la lib. Pas de `score()` | sklearn met les échantillons en lignes (axe 0) ; D2 met le temps en dernier axe. Sans clause explicite, « faisons comme sklearn » ferait rentrer la mauvaise convention par la fenêtre. Et `score()` est un « plus grand est meilleur », alors que BER/SER/EVM/MSE sont des « plus petit est meilleur » | Adoption intégrale de l'API sklearn | **Acté** |

**Restriction sur `__init__` (corollaire de D22).** sklearn interdit tout
calcul dans le constructeur pour permettre `clone()`. On retient une
version plus faible, adaptée : **pas de travail dépendant des données et
pas de création de RNG dans `__init__`/`__post_init__`** — cette seconde
clause est ce dont D6 a besoin pour re-semer les blocs après
construction. Le précalcul purement paramétrique (tables de treillis,
rayons CMA, masques d'allocation) reste autorisé et souhaitable.

**Trois régimes d'estimation**, à nommer explicitement dans chaque bloc :

- *Par bloc* — chaque trame porte ses pilotes ; `forward` estime et
  corrige sur la même donnée. Sémantique `fit_transform`, à assumer.
- *Préambule réutilisé* — `fit` une fois, `forward` N fois. Piloté par un
  drapeau `should_fit` uniformisé (aujourd'hui présent sur une seule
  classe).
- *Adaptatif* — l'estimée évolue pendant le traitement : `partial_fit`.

### 4.4 Rendu graphique (nouveau)

| # | Décision | Motif | Alternatives rejetées | Statut |
|---|----------|-------|-----------------------|--------|
| D25 | **Toute fonction ou méthode de tracé accepte `ax=None` et retourne `ax`.** Elle ne crée une figure que si `ax is None`, n'appelle jamais `plt.show()` dans `src/`, et le paramètre `num` (numéro de figure) disparaît. Corollaire : tout `plot_x()` est une couche mince au-dessus d'un `compute_x()` public retournant des tableaux | Aujourd'hui les fonctions *possèdent* la figure : impossible de superposer la courbe théorique, de juxtaposer deux constellations, ou d'exporter en PDF vectoriel pour un article. `num` est de la gestion d'état global matplotlib, incompatible avec un usage en script ou en notebook. La séparation calcul/rendu rend en outre le choix du backend indifférent côté utilisateur | Garder `num` ; fonctions de tracé autonomes appelant `show()` | **Acté** |
| D26 | **Un seul backend dans le cœur : matplotlib.** Les backends alternatifs (plotly, bokeh) vivent dans un paquet tiers ou, à la rigueur, un module optionnel `comnumpy.viz.plotly` importé paresseusement et déclaré en extra — **jamais** dans `dependencies` | Deux backends embarqués doublent la surface de test et de documentation, et l'abstraction fuit dès la première personnalisation. Deux asymétries tranchent pour matplotlib : les figures **vectorielles** du papier JOSS, et le fait que `kaleido` (export statique de plotly) **ne fonctionne pas dans Pyodide** — on perdrait à la fois le papier et la démo navigateur. Précédent : pandas définit un protocole de backend, que plotly implémente de son côté ; scikit-learn, statsmodels, librosa et xarray s'en tiennent à matplotlib | Registre de backends interne à la lib ; migration complète vers plotly | **Acté** |
| D27 | **Couleurs et style sont deux mécanismes séparés.** (a) Les couleurs *sémantiques* sont une table **gelée attachée au concept** — `CARRIER_STYLE` indexée par `CarrierType`, portant couleur, glyphe ASCII et libellé. (b) Le style des figures est une feuille `comnumpy.mplstyle` **livrée mais jamais appliquée à l'import**, activée explicitement. (c) Palette sûre pour le daltonisme, et **aucune information codée par la seule couleur** : marqueur ou hachure redondants | Une seule table pour la figure et la carte ASCII de D21 : les deux vues ne peuvent plus diverger (P7). Aujourd'hui elles divergent déjà — la signature dit `["b","g","r"]`, la docstring `["g","b","r","k"]`, et comme l'index porte le sens, la légende annonce l'inverse de ce que montre la figure. Importer une lib ne doit pas modifier l'état matplotlib de l'utilisateur : `core/visualizers.py` mute `rcParams` au niveau module, et seaborn a fait marche arrière sur exactement ce point. Public étudiant : `["b","g","r"]` est la combinaison la plus défavorable en vision des couleurs, et la redondance marqueur/hachure sauve aussi l'impression en niveaux de gris | Singleton mutable `comnumpy.config.colors` (état global, ordre-dépendant, non testable) ; application du style à l'import ; `rcParams` modifiés par la lib | **Acté** |

**Trois chemins de rendu, par ordre de robustesse.** Ils ne se
concurrencent pas, ils couvrent des contextes disjoints :

1. **ASCII** (D21b) — terminal, log, message d'erreur, doctest, CI. Aucune
   dépendance. C'est le seul qui fonctionne partout.
2. **matplotlib** — figures de la doc, des exemples, de `validation/` et
   du papier. Vectoriel, contrôlable, présent dans Pyodide.
3. **Interactif** — hors cœur, à la charge de l'utilisateur à partir des
   fonctions `compute_x()`.

**Gabarit normatif :**

```python
def plot_alphabet(alphabet, ax=None, label="alphabet", **kwargs):
    r"""Trace une constellation.

    Returns
    -------
    ax : matplotlib.axes.Axes
        L'axe utilisé, pour composition ultérieure.
    """
    import matplotlib.pyplot as plt        # local import (see D26)
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(alphabet.real, alphabet.imag, "o", label=label, **kwargs)
    ax.set_xlabel("In-phase"); ax.set_ylabel("Quadrature")
    return ax                              # never plt.show()
```

**Table sémantique (D27a).** Source unique de la couleur, du glyphe ASCII
et du libellé — la figure matplotlib et la carte spectrale de D21 lisent
la même ligne :

```python
CARRIER_STYLE = {                          # frozen, not globally configurable
    CarrierType.NULL:  {"color": "#BBBBBB", "glyph": ".", "hatch": "",   "label": "null"},
    CarrierType.DATA:  {"color": "#0072B2", "glyph": "#", "hatch": "",   "label": "data"},
    CarrierType.PILOT: {"color": "#D55E00", "glyph": "P", "hatch": "//", "label": "pilote"},
}
```

Couleurs issues de la palette Okabe-Ito. Le champ `hatch` porte la
redondance exigée par D27c : la distinction reste lisible sans la
couleur. Les signatures de tracé conservent `color=None` avec repli sur
la table, pour que l'utilisateur garde le dernier mot (D25).

**Feuille de style (D27b).** `comnumpy.mplstyle` est livrée dans le
paquet et appliquée explicitement — par les scripts `validation/`, la
construction de la doc et les figures du papier, jamais à l'import :

```python
plt.style.use(comnumpy.style.PATH)         # global, explicit
with comnumpy.style.context():             # or scoped
    ...
```

### 4.5 Structure de trame (nouveau)

| # | Décision | Motif | Alternatives rejetées | Statut |
|---|----------|-------|-----------------------|--------|
| D28 | Une trame est décrite par un objet `FrameStructure` (dataclass **frozen**) portant une **liste ordonnée de champs typés** `FrameField(name, role, values)`, avec `role ∈ FieldRole {SYNC, TRAINING, HEADER, PAYLOAD, TAIL, PAD}`. `Framer` et `Deframer` partagent le même objet. Un champ dont `values is None` est inconnu à l'émission | Un préambule unique ne suffit pas : 802.11 sépare STF et LTF parce que les rôles diffèrent — le STF sert à l'AGC et à la synchronisation grossière, le LTF à l'estimation de canal — et les critères de conception divergent (autocorrélation piquée contre spectre plat). Avec des champs typés, ajouter une partie devient **une ligne dans la structure**, pas un paramètre de plus sur chaque bloc. C'est le patron de D15 appliqué à l'autre axe : `CarrierType` type l'axe fréquence, `FieldRole` type l'axe temps. Par ailleurs la moitié réceptrice existe déjà (`DataAidedSimpleSynchronizer.fit(x, x_preamble)`) sans qu'aucun bloc émetteur n'insère quoi que ce soit : même classe de bugs « configurations divergentes » que D18 | Champ `preamble` unique ; fusion de `CarrierAllocation` et `FrameStructure` en un seul objet (sur-abstraction : les deux axes n'ont pas les mêmes invariants) | **Acté** |
| D29 | Catalogue de séquences de synchronisation en fonctions libres : `zadoff_chu`, `schmidl_cox_preamble`, `barker`, `golay_pair`, `m_sequence`. Chacune porte sa clause de référence et son test de propriété (autocorrélation, amplitude constante) | Ce sont toujours les mêmes, dans tous les standards : Zadoff-Chu/CAZAC est le choix moderne (PRACH LTE, NR), Schmidl-Cox est le cas d'école OFDM qui donne timing *et* CFO, Barker relève de 802.11b et Golay de 802.11ad. Les livrer évite que chaque utilisateur les réimplémente avec une convention d'indice différente | Laisser l'utilisateur fournir sa séquence uniquement ; classes par séquence | **Acté** |

**Où placer l'estimation de canal (clause normative).** Trois régimes,
trois emplacements. Ne pas mettre dans la trame ce qui relève de
l'allocation :

| Régime | Emplacement | Cycle de vie (D22) |
|---|---|---|
| Pilotes dispersés dans la grille (CRS LTE, DM-RS NR, DVB-T) | `CarrierAllocation` (D15), porteuses `PILOT` | par bloc — `forward` estime et corrige |
| Champ d'apprentissage en tête de trame (LTF 802.11, *training sequence* cohérente) | `FrameStructure`, `FieldRole.TRAINING` | `fit()` une fois, `forward()` N fois |
| Midamble, ou suivi décision-dirigé | champ inséré, ou aucun | `partial_fit()` |

**Clause d'axes (corollaire de D2) — *corrigée*.** Le `Framer` est un
bloc de **layout Bloc `(..., T, F)` dans les deux cas**. Ce qui change
n'est pas le layout, c'est la signification de `T` :

| | `T` désigne | `F` désigne | Les champs s'ajoutent sur | Indice de trame |
|---|---|---|---|---|
| **Mono-porteuse** | l'indice de trame | la position dans la trame | **`F`** (axe −1) | `T` lui-même |
| **OFDM** | le symbole OFDM dans la trame | la sous-porteuse | **`T`** (axe −2) | axe de tête (batch) |

En OFDM les champs sont des **symboles OFDM entiers** — c'est le
`sync_words` de GNU Radio — et l'indice de trame remonte dans les axes de
tête, ce qui est conforme au §2 : *le batch n'est jamais un axe nommé*.
Un lot de trames est un batch, aucun axe structurel supplémentaire n'est
nécessaire.

**Position dans la chaîne.** Le `Framer` est toujours **entre le S/P et
le P/S**, mais l'encadrement n'est serré qu'en mono-porteuse :

```
mono-porteuse :  S/P -> Framer -> P/S
OFDM          :  S/P -> CarrierAllocator -> Framer -> IFFT -> CP -> P/S
```

**Couplage avec le S/P (normatif).** La taille de bloc du S/P n'est
**pas** un paramètre libre : elle vaut `frame.payload_length`. Aujourd'hui
`Serial2Parallel(N_sub=...)` se règle à la main, donc un S/P à 1000 face
à une structure à 1024 produit un remplissage silencieux ou une exception
cent lignes plus loin, selon la méthode de *padding* — c'est la classe de
bugs « configurations divergentes » de D18, sur un autre axe.
`Framer.prepare()` la ferme, conformément à sa catégorie *Axe déclaré* :

```
ValueError: Framer expects (..., T, 1024) for frame '802.11a' (payload
field length 1024), got (..., T, 1000). Set Serial2Parallel(N_sub=1024)
or use frame.payload_length.
```

**Lien avec D22.** Un champ `TRAINING` **est** le `y` de `fit(X, y)`.
L'estimateur désigne le champ par son nom, et `fit()` récupère du même
coup les échantillons reçus à la bonne position *et* les valeurs émises
connues :

```python
ChannelEstimator(frame, field="LTF")
```

Plus aucune ambiguïté sur ce qu'est le second argument — c'est le défaut
relevé à l'audit, où il désigne tour à tour `x_target`, `x_preamble` et
`w0`, poussé jusqu'à sa résolution.

```python
class FieldRole(IntEnum):
    SYNC = 0         # detection, AGC, timing, coarse CFO
    TRAINING = 1     # channel estimation
    HEADER = 2
    PAYLOAD = 3
    TAIL = 4
    PAD = 5

@dataclass(frozen=True)
class FrameField:
    name: str                        # "STF", "LTF", "SIG", ...
    role: FieldRole
    values: Optional[np.ndarray]     # None = unknown at transmit time

@dataclass(frozen=True)
class FrameStructure:
    fields: Tuple[FrameField, ...]
    standard: str = "custom"
    reference: str = ""              # clause of the standard
    # invariants: exactly one PAYLOAD field; every other field carries values
    # properties: frame_length, slice_of(name), fields_by_role()

class Framer(Processor):    ...      # (..., N_payload) -> (..., N_frame)
class Deframer(Processor):  ...      # inverse; exposes the extracted fields
```

**Rendu (P7).** Une structure de trame se lit comme une allocation se
voit — même mécanisme que la carte spectrale de D21, autre axe :

```
>>> print(get_frame("802.11a", payload=1024))
802.11a PPDU                        [IEEE 802.11-2020, 17.3.2]
|--STF--|--LTF--|SIG|-------------- PAYLOAD --------------|tail|pad|
   160     160   80              8192                        6   26
  sync   training                                    unknown at TX
```

### 4.6 Langue (nouveau)

| # | Décision | Motif | Alternatives rejetées | Statut |
|---|----------|-------|-----------------------|--------|
| D30 | **Tout ce qui est livré est en anglais** : identifiants, docstrings, commentaires de code, messages d'erreur et de `prepare()`, documentation, exemples, `validation/`, README, CHANGELOG. Les documents de conception internes — le présent ADD — restent en français, et c'est la seule exception | Public international et cible JOSS : un reviewer, un contributeur ou un étudiant étranger doit lire le code sans traduction. Un commentaire français dans une docstring publiée est un obstacle gratuit à la contribution. La frontière est nette et vérifiable en revue : ce qui est distribué dans le paquet ou publié sur le site est en anglais, le reste ne l'est pas | Bilinguisme (charge de maintenance doublée, désynchronisation garantie) ; tout en français (exclut la contribution externe et la soumission JOSS) | **Acté** |

Cette règle s'applique **aussi aux extraits de code du présent
document** : les commentaires des esquisses sont en anglais, parce
qu'ils sont destinés à être copiés dans `src/`. La prose d'architecture,
elle, reste en français.

### 4.7 Introspection et sérialisation (nouveau)

| # | Décision | Motif | Alternatives rejetées | Statut |
|---|----------|-------|-----------------------|--------|
| D31 | **Une chaîne s'exporte et se relit en JSON.** Chaque bloc est `{"id", "type", "params", "inputs"}`. On sérialise l'**intention** (`{"type": "QAM", "M": 16}`), jamais la donnée développée (les 16 complexes). Les tableaux véritablement libres partent dans un `.npz` associé. Le champ `inputs` est présent dès la v1.0 même s'il est toujours implicite | `Sequential` + dataclasses **sont déjà le schéma** : `asdict()` produit les paramètres, `dataclasses.fields()` donne noms, types et défauts, soit un formulaire d'UX gratuit. Il ne manque que le type du bloc — rien ne dit aujourd'hui que `id2` est un `AWGN` — et un traitement des tableaux, puisque `json.dumps(chain.asdict())` échoue sur `ndarray`. Sérialiser l'intention rend le fichier lisible, éditable et robuste au changement d'implémentation ; c'est déjà la logique de D15/D17, où `get_allocation("802.11a")` est sérialisable là où le masque ne l'est pas. `inputs` ne coûte rien maintenant et évite un format v2 incompatible le jour où une branche apparaît (cf. §8) | Liste ordonnée nue sans `id` (ferme la porte au DAG) ; `pickle` (non lisible, non sûr, lié à la version) ; tableaux développés en JSON | **Acté** |
| D32 | **Test d'aller-retour normatif**, paramétré sur le catalogue d'exemples : `from_json(to_json(chain))` re-exécuté à graine égale redonne le **même signal**. Condition de merge de toute évolution du format | Un seul test couvre l'ensemble de la chaîne de sérialisation — types, paramètres, tableaux, ordre. Couplé à D6, il transforme « reproduisez cette figure » en fichier joint plutôt qu'en promesse, ce qui est un argument JOSS direct | Tests par bloc (ne détectent pas les erreurs de composition) | **Acté** |
| D33 | **Trois vues d'introspection**, sans dépendance ajoutée : (a) `__repr__` structurel emprunté à `torch.nn.Module` ; (b) `chain.summary(N)` tabulant bloc, forme de sortie, dtype et temps, emprunté à `torchinfo` ; (c) `chain.to_mermaid()`, rendu par `sphinxcontrib-mermaid` déjà présent dans les extras | Les bibliothèques PyTorch elles-mêmes sont inutilisables — elles tracent des `nn.Module` avec des tenseurs torch, et `torchviz` dessine le graphe autograd, inexistant ici. Leurs **idées** se transposent en une quinzaine de lignes chacune. `summary()` rend en outre la convention d'axes de D2 **visible** (`(4096,)` → `(64, 64)` → `(4096,)`), ce qui en fait l'outil de débogage naturel du refactor et un support de cours. La machinerie existe déjà : `profile_execution_time` fait le même parcours | Export ONNX pour réutiliser Netron (format d'opérateurs ML, mauvais ajustement) ; dépendance à graphviz | **Acté** |

**Format (D31).** L'exemple minimal ; `inputs` est omissible et vaut par
défaut le bloc précédent :

```json
{
  "comnumpy": "1.0",
  "blocks": [
    {"id": "gen",  "type": "SymbolGenerator", "params": {"M": 16, "seed": 42}},
    {"id": "map",  "type": "SymbolMapper",
     "params": {"alphabet": {"type": "QAM", "M": 16}}, "inputs": ["gen"]},
    {"id": "awgn", "type": "AWGN",
     "params": {"snr_dB": 15}, "inputs": ["map"]}    // D41: one kwarg per parameterization
  ]
}
```

Le registre nom → classe réutilise le mécanisme du décorateur
`register_allocation` de D17 : écrit une fois, utilisé deux fois.

**Limite assumée.** Les *callables* — callbacks, la `rule` de
`scattered_allocation` — ne sont pas sérialisables. À documenter comme
frontière, pas à contourner.

**Vues (D33).** Sorties réelles, prototypées :

```
>>> chain
Sequential(
  (0): SymbolGenerator(M=16, seed=1)
  (1): SymbolMapper(alphabet=ndarray(16,))
  (2): Serial2Parallel(N_sub=64, order='F', method='zero-padding')
  (3): AWGN(snr_dB=15, seed=2)
)

>>> chain.summary(4096)
#    block               name              output shape    dtype         time ms
---  ------------------  ----------------  --------------  ------------  --------
0    SymbolGenerator     generator         (4096,)         int64            0.11
1    SymbolMapper        Symbol Mapper     (4096,)         complex128       0.04
2    Serial2Parallel     S2P               (64, 64)        complex128       0.01
3    AWGN                awgn              (4096,)         complex128       0.79
```

### 4.8 Exécution de simulations (nouveau)

| # | Décision | Motif | Alternatives rejetées | Statut |
|---|----------|-------|-----------------------|--------|
| D34 | **Toute chaîne est reconfigurable après construction** : `chain.set_params(**{"awgn.value": 12})` modifie le paramètre désigné en notation pointée `id_bloc.champ` et redéclenche le précalcul paramétrique du bloc (`__post_init__` ou équivalent). Emprunt du `set_params` de scikit-learn, adressage par les `id` de D31. Corollaire rendu possible par la restriction sur `__init__` de la §4.3 : puisque le constructeur ne fait ni travail dépendant des données ni création de RNG, reconstruire l'état d'un bloc après modification d'un champ est sûr et bon marché | Tout balayage de paramètre (BER vs SNR, pénalité vs distance) exige de modifier un paramètre et relancer. Aujourd'hui la seule voie est de reconstruire la chaîne entière à la main, en dupliquant sa définition — la classe de bugs « configurations divergentes » de D18, appliquée au temps : la chaîne du point de mesure n s'écarte silencieusement de celle du point 0. `set_params` + `Sequential.seed` (D6) forment le couple minimal : reconfigurer, re-semer, relancer | Mutation directe des attributs (saute le précalcul → état incohérent, exactement le piège que D23 nomme) ; reconstruction depuis le JSON D31 avec surcharge (valide, mais plus coûteux à écrire pour l'usage interactif ; reste la voie recommandée pour les scénarios sérialisés) | **Acté** |
| D35 | **Le balayage est une fonction, pas un framework.** Forme cible : `sweep(chain, param, values, metrics, seed, n_frames)` retournant un dict de tableaux — le patron `GridSearchCV`/`cross_validate` de sklearn, cohérent avec l'emprunt D22–D24. **L'implémentation est différée** : elle n'entre dans `src/` qu'après que trois scripts `validation/` (D7) auront exhibé le même squelette recopié à l'identique, et sa forme finale sera extraite de ces scripts, pas conçue a priori. Le patron *Trainer* de PyTorch Lightning (inversion de contrôle par hooks) est **explicitement rejeté** | Un balayage Monte-Carlo est une boucle `for` sans état, sans optimiseur, embarrassingly parallel : l'inversion de contrôle rendrait la boucle invisible, ce qui contredit P2 frontalement — un étudiant doit pouvoir lire *où* le SNR est balayé. À l'inverse, tout coder à la main pour chaque figure recopie le triplet construire/balayer/collecter. La fonction sklearn-like est le point d'équilibre : la boucle reste visible dans son implémentation (quinze lignes lisibles), l'appelant n'écrit qu'une ligne. D31/D32/D6 fournissent déjà sérialisation et reproductibilité ; un « scénario » n'est que chaîne JSON + spec de balayage + graine + noms de métriques — un format de fichier avant d'être du code | *Trainer* à hooks (inversion de contrôle injustifiée, boucle invisible) ; classe `Simulation` avec cycle de vie (sur-abstraction pour une boucle sans état) ; implémentation immédiate (violerait le principe de complexité à la demande : la forme doit sortir d'un motif constaté dans `validation/`, pas d'une anticipation) | **Acté — implémentation différée, déclencheur : 3ᵉ script `validation/` au squelette identique** |

**Esquisse (D35), non normative tant que le déclencheur n'est pas atteint :**

```python
results = sweep(
    chain,
    param="awgn.snr_dB",                 # dotted path, resolved by D34
    values=np.arange(0, 21, 2),
    metrics={"ser": compute_ser},        # smaller-is-better, no score() (D24)
    seed=42,                             # per-point reseed via D6
)
# results["ser"] -> ndarray aligned with values
```

**Frontières.** (a) Parallélisation : hors périmètre v1.0 — la boucle est
séquentielle et lisible ; `concurrent.futures` viendra si un usage réel
le demande, jamais une dépendance. (b) *Loggers* interchangeables
(console, CSV, `.npz`) : idée retenue de Lightning, différée au même
déclencheur que D35. (c) Balayage multi-paramètres (grille) : `param`
et `values` acceptent des listes ; produit cartésien hors périmètre tant
qu'un cas réel ne l'exige pas.

### 4.9 Ergonomie d'API et qualité de code (nouveau)

Principe directeur de cette section : **la simplicité d'usage se mesure,
elle ne se déclare pas.** Chaque décision porte donc un critère
vérifiable en CI, pas une intention.

| # | Décision | Motif | Alternatives rejetées | Statut |
|---|----------|-------|-----------------------|--------|
| D36 | **Surface publique plate et explicite.** Tout ce dont un utilisateur a besoin s'importe depuis `comnumpy` ou un sous-paquet de premier niveau (`comnumpy.ofdm`, `comnumpy.mimo`, `comnumpy.optical`, `comnumpy.fec`). Chaque module public déclare `__all__` ; les modules d'implémentation sont préfixés `_` et hors contrat. matplotlib n'est **jamais** importé par `import comnumpy` : les fonctions de tracé l'importent localement (déjà le gabarit D25), et le sous-module de style est chargé paresseusement (PEP 562, `__getattr__` de module). Deux budgets vérifiés en CI : `import comnumpy` < 200 ms, et `"matplotlib" not in sys.modules` après cet import | Le B1 de l'audit vient en partie de là : le README invente une API parce qu'aucune surface publique n'est définie — ce que l'auteur lui-même croit exporter n'est écrit nulle part. Une API plate est aussi ce qui rend lisibles la doc, les messages d'erreur et le champ `type` du JSON D31. L'import léger sert Pyodide (démarrage du notebook navigateur) et les scripts `rx.py` du BANC qui n'ont pas besoin de figures | Imports profonds obligatoires (`comnumpy.core.processors.AWGN` — fige l'arborescence interne dans l'API publique, exactement ce qu'un refactor D2 ne peut pas se permettre) ; tout réexporter à la racine (namespace pollué, autocomplétion inutilisable) | **Acté** |
| D37 | **Typage statique livré, pas décoratif.** Annotations complètes sur toute l'API publique, marqueur `py.typed` (PEP 561) dans le paquet, vérification `pyright` en mode strict sur `src/`, bloquante en CI. Tableaux : `npt.ArrayLike` en entrée (on accepte listes et scalaires), `npt.NDArray[np.complex128]` en sortie quand le dtype est contractuel. `from __future__ import annotations` partout ; syntaxe 3.11+ native (`X | None`, `Self`) | L'autocomplétion et l'erreur *à l'écriture* sont la moitié de la simplicité perçue d'une lib. Et le typage attrape des classes entières de défauts constatés : B5 (`extractor: field(...)` — un appel dans l'emplacement du type) est signalé par n'importe quel vérificateur en une seconde, de même que `target_data = Union[...]` (attribut de classe au lieu d'un champ). Le typage documente en outre les contrats que les shapes ne peuvent pas dire — quel bloc accepte du réel, lequel exige du complexe | Stubs `.pyi` séparés (désynchronisation garantie pour un mainteneur seul) ; typage « au besoin » (jamais terminé, et un `py.typed` partiel est pire que pas de `py.typed` : pyright fait alors confiance à des trous) | **Acté** |
| D38 | **Hiérarchie d'exceptions à trois membres, pas plus** : `ComnumpyError(Exception)` racine ; `ShapeError(ComnumpyError, ValueError)` pour tout refus de `prepare()` ; `NotFittedError(ComnumpyError, RuntimeError)` pour D23. Le double héritage préserve les `except ValueError` existants. Le gabarit de message de la §8 devient **normatif** : *constat, attendu, action* — « got X, expected Y, do Z ». Jamais d'`assert` pour la validation d'entrée utilisateur (`python -O` les supprime) | P5 exige de refuser explicitement ; ceci outille le refus. `except ComnumpyError` sépare « j'ai mal configuré ma chaîne » de « bug numpy sous-jacent », distinction qu'un `ValueError` nu ne fait pas. Trois classes suffisent : chaque exception supplémentaire est une décision de plus imposée à l'utilisateur au moment du `except` | Une exception par module (taxonomie sans usage) ; codes d'erreur ; `ValueError` nus (état actuel — indistinguables de numpy) | **Acté** |
| D39 | **Un seul outil de qualité : `ruff`** (lint **et** format, remplace flake8 + black + isort), configuré dans `pyproject.toml`, exécuté par pre-commit et en CI. Couverture mesurée avec **seuil cliquet** : le pourcentage constaté devient le minimum exigé, il ne peut que monter. Complète C1–C5 : matrice 3.11/3.12/3.13, doctests bloquants (D8), étape `pip install .` + import nu | Zéro débat de style pour un projet à mainteneur unique qui espère des contributions JOSS : le formateur tranche, la revue parle d'architecture. Le cliquet évite le double piège des seuils : l'objectif arbitraire (90 % ?) et les tests de remplissage écrits pour l'atteindre — on interdit seulement la régression, ce qui est le vrai risque constaté (`optical/` à 0 %) | black + flake8 + isort (trois outils, trois configs, conflits connus) ; seuil de couverture fixe ; pylint (lent, verbeux) | **Acté** |
| D40 | **Python moderne comme filet, budget de simplicité comme contrat.** (a) Toutes les dataclasses passent en `slots=True` ; les paramètres optionnels sont `kw_only=True` ; les objets valeur (`CarrierAllocation`, `FrameStructure`, `FrameField`) restent `frozen`. (b) Dans toute signature publique, seul le premier argument (la donnée ou le paramètre principal) est positionnel ; le reste est keyword-only. (c) **Budget d'ergonomie normatif, vérifié par doctest** : l'exemple du README tient en ≤ 8 lignes imports compris ; aucune tâche du tutoriel n'importe un module préfixé `_` ; toute chaîne de la doc se construit sans variable intermédiaire autre que les blocs eux-mêmes | `slots` transforme la classe de défauts S1/S2 (typo d'attribut, `__post__init__` jamais appelé silencieusement) d'un bug silencieux en `AttributeError` immédiate — c'est le même mouvement que P5, appliqué à la lib elle-même. Le kw-only rend les appels auto-documentés (`AWGN(snr_dB=15)` se lit sans la doc, P7) et rend l'ajout de paramètres non cassant. Le budget (c) rend « simple à utiliser » **falsifiable** : si l'exemple ne tient plus en 8 lignes, c'est l'API qui a tort, pas l'exemple | `attrs` (dépendance, viole P1) ; `pydantic` (validation runtime coûteuse sur des tableaux, dépendance lourde) ; budget en prose non testé (vœu pieux — exactement ce que P3 interdit ailleurs) | **Acté** |
| D41 | **Paramétrages alternatifs : un kwarg nommé par paramétrage, mutuellement exclusifs — jamais de couple `value`/`unit`.** Un bloc admettant plusieurs façons équivalentes d'être configuré expose un keyword par voie (`AWGN(snr_dB=15)` **ou** `AWGN(sigma2=0.01)`), tous `None` par défaut, avec validation « exactement un » dans `__post_init__` et message énumérant les choix. Le nom du kwarg porte l'unité (`snr_dB`, `sigma2`, `length_km`) : il n'existe **aucun** paramètre `unit` dans la lib | Le couple `value`/`unit` actuel est de la *primitive obsession* : deux paramètres sans sens séparés, où la faute de frappe (`"snr_db"`) traverse jusqu'au runtime. Surtout, `snr_dB` et `sigma2` ne sont pas la même grandeur dans deux unités — l'un est relatif à la puissance du signal, l'autre absolu ; ils changent le *calcul* de `forward`, pas une conversion. Un kwarg par voie rend la sémantique visible dans l'appel (P7), la faute de frappe devient `TypeError` immédiate (D40a/b), le JSON D31 sérialise `{"snr_dB": 15}` sans champ discriminant, et `sweep(param="awgn.snr_dB")` (D35) devient auto-descriptif là où `awgn.value` était muet. Précédent : `timedelta(hours=…, minutes=…)` | `Literal["snr_dB", "sigma2"]` sur `unit` (corrige le typage, garde le couple, masque la différence relatif/absolu) ; `IntEnum NoiseUnit` (verbeux à l'appel, un import de plus, conversion JSON nécessaire — les enums D15/D21 se justifient pour des *données*, pas un sélecteur binaire) ; `pint`/objets quantité (viole P1) ; surcharge par type (indécidable : tout est `float`) | **Acté** |

**Piège d'implémentation (D40a, à connaître avant le refactor).** Avec
`slots=True`, tout attribut créé dans `__post_init__` doit être déclaré
comme champ `field(init=False)` — sinon `AttributeError` à la
construction. Concerne les tables précalculées (treillis D4 :
`next_state`, `outputs`) et les grandeurs estimées de D23 (`theta_`,
`H_`), qui se déclarent :

```python
@dataclass(slots=True)
class ViterbiDecoder(Processor):
    g: tuple[int, ...] = (0o133, 0o171)
    K: int = field(default=7, kw_only=True)
    soft: bool = field(default=False, kw_only=True)
    # precomputed tables (parametric, allowed in __post_init__ per §4.3)
    next_state: np.ndarray = field(init=False, repr=False)
    outputs: np.ndarray = field(init=False, repr=False)
```

C'est une contrainte, mais c'est la bonne : la liste des champs devient
le contrat exhaustif de l'état du bloc, ce qui est précisément ce que
D31 sérialise et ce que `__repr__` (D33a) montre.

**Gabarit (D41), normatif.** La validation vit dans `__post_init__`
(purement paramétrique, autorisée par la §4.3) ; le message énumère les
voies :

```python
@dataclass(slots=True)
class AWGN(Processor):
    snr_dB: float | None = field(default=None, kw_only=True)
    sigma2: float | None = field(default=None, kw_only=True)
    seed: int | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        if (self.snr_dB is None) == (self.sigma2 is None):
            raise ValueError(
                "AWGN: specify exactly one of snr_dB= (relative to "
                "measured signal power) or sigma2= (absolute noise "
                "variance); got both or neither."
            )
```

**Clause de frontière (D41).** `esn0_dB` et `ebn0_dB` **n'entrent pas**
dans `AWGN` : les convertir en variance exige les bits par symbole et le
rendement de code, qui sont des connaissances de *chaîne*, pas de bloc —
les y admettre recréerait la classe de bugs « configurations
divergentes » de D18 (un `AWGN(ebn0_dB=…)` réglé pour du 16-QAM codé
resterait silencieusement faux après passage en QPSK). La conversion se
fait côté script, où ces grandeurs sont connues, avec des utilitaires
libres `ebn0_to_snr_dB(ebn0_dB, bits_per_symbol, code_rate)` fournis et
doctestés. Même statut que les limites documentées du module Raman :
frontière explicite, pas contournement.

**Migration.** `AWGN(value=…, unit=…)` disparaît dans la fenêtre de
casse du jalon 1 (avec D2/D36/D38/D40) et figure dans la table de
migration du CHANGELOG. Le motif s'applique à tout bloc à paramétrages
multiples — atténuation en dB/km ou linéaire, durées en échantillons ou
en secondes — au fur et à mesure qu'ils sont touchés.

**Cible d'usage (D36 + D40c réunis).** L'exemple canonique, celui du
README et du doctest bloquant D8, dans sa forme budgétée :

```python
from comnumpy import Sequential, SymbolGenerator, SymbolMapper, \
    SymbolDemapper, AWGN, get_alphabet, compute_ser

alphabet = get_alphabet("QAM", 16)
chain = Sequential([SymbolGenerator(M=16, name="tx"), SymbolMapper(alphabet),
                    AWGN(snr_dB=15), SymbolDemapper(alphabet)], taps=["tx"])
chain.seed(42)                       # D6 seeded, D42 records "tx"
detected = chain(10_000)
ser = compute_ser(chain.tap("tx"), detected)   # SER = 0.0165
```

Huit lignes, un seul import, aucune connaissance de l'arborescence
interne. Si une décision future rend cet exemple plus long, elle amende
D40c explicitement ou elle est refusée.

*Mis à jour v0.5.* D42 retire le `Recorder` de la liste : la chaîne ne
décrit plus que le système, à budget constant (le `rec = Recorder()` est
remplacé par la ligne d'exécution). La version v0.4 de cet exemple
condensait exécution et mesure en `compute_ser(rec.get_data(),
chain(10_000))` — **qui ne pouvait pas fonctionner** : Python évalue ses
arguments de gauche à droite, donc la lecture précédait l'exécution et
levait sur un enregistrement vide. Le défaut a survécu à deux versions de
l'ADD parce que l'exemple n'avait jamais été exécuté — soit précisément
ce que P3 et D8 existent pour attraper, dans le document qui les édicte.
L'exemple ci-dessus est exécuté ; le SER en commentaire est celui qu'il
imprime.

### 4.10 Docstring comme support de cours (amendement de D10)

La docstring d'un bloc n'est pas un commentaire : c'est **la page de
cours du concept qu'il implémente**. Un étudiant qui fait
`help(AWGN)` ou ouvre la page Sphinx doit y trouver, dans cet ordre : le
modèle mathématique, la correspondance exacte entre chaque symbole de
l'équation et chaque paramètre du constructeur, la convention d'axes, et
un exemple exécutable dont les nombres sont vrais. Cinq règles
normatives :

**R1 — Section « Signal Model » en tête, en LaTeX numpydoc.** Toute
classe `Processor` et toute fonction de traitement l'exige. L'équation
utilise la notation de la table unique de CONVENTIONS.md (`x` émis, `y`
reçu, `n` temps discret, `k` indice de sous-porteuse, `\sigma^2`
variance de bruit…) — **une lettre, un sens, dans toute la lib**. Deux
blocs qui noteraient différemment la même grandeur sont un défaut, au
même titre que S11.

**R2 — Bijection symbole ↔ paramètre.** Chaque symbole libre de
l'équation apparaît dans `Parameters` (ou `Attributes` s'il est estimé,
convention D23) avec son `:math:` dans la description ; réciproquement,
chaque paramètre numérique cite son symbole. Un symbole sans paramètre
ou un paramètre sans symbole est une docstring incomplète. C'est cette
bijection qui fait le cours : l'étudiant lit l'équation, puis *voit* où
chaque grandeur se règle.

**R3 — Sections obligatoires et leur ordre** : résumé d'une ligne ·
`Signal Model` · catégorie d'axes (une ligne : *element-wise*,
*axis −1*, ou *declared axis* avec la forme exigée) · `Parameters` ·
`Attributes` (grandeurs estimées `_`, D23) · `Raises` (conditions de
`ShapeError`, D38) · `References` · `Examples`. La section `References`
cite le manuel (Proakis, Goldsmith…) ou la clause du standard quand
l'algorithme en sort — même exigence de provenance que D15/D20, appliquée
au texte.

**R4 — L'exemple est un doctest exécuté (D8), déterministe (graine
explicite), et ses nombres sont vrais (P3).** Jamais plus de dix lignes ;
s'il en faut plus, c'est un script `validation/`, pas un doctest.

**R5 — Vérification outillée, pas promise** (principe de la §4.9) : la
validation `numpydoc` (sections présentes, ordre, paramètres documentés =
signature) s'ajoute à la CI de D39, et les doctests bloquants de D8
vérifient R4. R1 et R2 restent en revue humaine — ils portent du sens,
pas de la forme.

**Gabarit normatif** (l'exemple canonique, à recopier) :

```python
@dataclass(slots=True)
class AWGN(Processor):
    r"""Additive white Gaussian noise channel.

    Signal Model
    ------------
    .. math::

        y[n] = x[n] + b[n], \qquad
        b[n] \sim \mathcal{CN}\left(0, \sigma^2\right)

    When parameterized by ``snr_dB``, the variance is derived from the
    measured input power :math:`P_x = \mathbb{E}\left[|x[n]|^2\right]`:

    .. math::

        \sigma^2 = P_x \, 10^{-\mathrm{SNR_{dB}}/10}

    Axes: *element-wise* — applied pointwise, shape-agnostic.

    Parameters
    ----------
    snr_dB : float, keyword-only
        Signal-to-noise ratio :math:`\mathrm{SNR_{dB}}` in decibels,
        relative to the measured input power. Mutually exclusive with
        ``sigma2``.
    sigma2 : float, keyword-only
        Absolute noise variance :math:`\sigma^2`. Mutually exclusive
        with ``snr_dB``.
    seed : int, optional, keyword-only
        Local RNG seed; overridden by chain seeding (``Sequential.seed``).

    Attributes
    ----------
    sigma2_ : float
        Variance actually applied. Estimated from the input power when
        parameterized by ``snr_dB`` (data-dependent, hence the trailing
        underscore); equal to ``sigma2`` otherwise.

    References
    ----------
    J. G. Proakis, M. Salehi, *Digital Communications*, 5th ed.,
    McGraw-Hill, 2008, Section 4.2.

    Examples
    --------
    >>> x = np.zeros(10_000, dtype=complex)
    >>> y = AWGN(sigma2=0.01, seed=42)(x)
    >>> round(np.var(y), 3)
    0.01
    """
```

Ce gabarit est le contrat : `Signal Model` donne le cours, la bijection
R2 relie chaque :math:`\sigma^2` à son champ, la ligne d'axes rend D2
visible, `sigma2_` illustre D23, le doctest prouve P3. Les pages Sphinx
générées depuis ces docstrings **sont** le support de cours — pas un
document parallèle à maintenir, le même texte, garanti juste parce
qu'exécuté.

### 4.11 Observation et câblage de chaîne (nouveau)

| # | Décision | Motif | Alternatives rejetées | Statut |
|---|----------|-------|-----------------------|--------|
| D42 | **`module_list` ne contient que le système de communication.** (a) Tout bloc dont la seule fonction est d'observer est **supprimé** — `Recorder`, `Logger`, `Debugger`, `PowerReporter`, `TimeSignalMonitor`, `MetricRecorder`, la famille `Scope`, `FFTMonitor` ; `core/monitors.py` disparaît. (b) L'observation devient une **métadonnée de chaîne** : `Sequential(..., taps=["id"])` enregistre la sortie des blocs nommés pendant `forward`, relue par `chain.tap("id")`. (c) Une arête de donnée supplémentaire se déclare de la même façon : `Sequential(..., wiring={"bloc.param": "source"})` affecte au bloc cible, avant son exécution, le signal produit par la source **dans la même passe**. (d) Le tracé et le compte rendu sont des **fonctions** appliquées aux tableaux extraits (`plot_iq`, `plot_spectrum`, …, `signal_report`), jamais des blocs. (e) Ce qu'un bloc calcule en interne et que sa sortie ne contient pas s'expose en attribut souligné D23 (`pilots_` de `CarrierExtractor`), pas par un puits externe | Un `Recorder` inséré dans `module_list` fait mentir toutes les vues qui décrivent la chaîne — `__repr__`, `summary()`, `to_mermaid()` (D33), l'export JSON et les indices de `set_params` (D31/D34) voient un système à N+2 blocs. La ligne de partage est nette : un tap capture ce qui **circule sur le fil**, un attribut souligné expose ce qu'un bloc a calculé et que sa sortie ne porte pas. Le coût est une écriture de dictionnaire par bloc tappé — une **référence**, sans copie ; ce qui repose sur l'invariant, désormais écrit dans CONVENTIONS.md, qu'aucun bloc ne modifie son entrée sur place. (c) traite le seul cas que (b) ne couvre pas et que le `Recorder` servait vraiment : un estimateur assisté par les données dont la référence est **produite par la chaîne**. Un `reference=` figé y est faux et **silencieusement** faux — en Monte-Carlo il continue de comparer aux symboles du premier tirage. C'est la forme bornée du champ `inputs` de D31 : une seconde entrée déclarée par la chaîne, sans DAG général (§8) | Filtrer les `Recorder` à l'affichage (le mensonge change de place : sérialisation, indices et `set_params` les voient toujours) ; mode « tout tracer » (coût mémoire réel sur les chaînes optiques full-field, contraire à P2) ; laisser les blocs se référencer entre eux comme avant (arête de graphe passant par la configuration — non sérialisable : un paramètre *callable* est une frontière assumée de D31, donc toute chaîne enregistrant ses pilotes était inexportable) ; hooks à la PyTorch (machinerie invisible, contraire à P7) | **Acté** |

**Trois besoins, trois mécanismes.** La distinction est normative ; toute
demande d'observation doit tomber dans l'une des trois cases.

| Besoin | Mécanisme |
|---|---|
| Regarder un signal après coup | `taps=[...]` puis `chain.tap(id)` |
| Alimenter un bloc avec un signal amont **du tirage courant** | `wiring={"bloc.param": "source"}` |
| Référence connue d'avance (préambule, séquence d'apprentissage) | paramètre tableau du bloc |

**Forme (b) et (c) réunies**, exécutée — un compensateur de phase assisté
par les données, dont la référence est produite par la chaîne elle-même :

```python
chain = Sequential([
    SymbolGenerator(16, name="data_tx"),
    SymbolMapper(alphabet, name="signal_tx"),
    CFO(0.001), IQImbalance(*iq), AWGN(sigma2=0.005, name="awgn"),
    BlindIQCompensator(name="gsop"), BlindCFOCompensator(name="cfo"),
    DataAidedPhaseCompensator(reference=np.zeros(1), name="phase"),
    SymbolDemapper(alphabet),
], taps=["data_tx", "awgn", "gsop", "phase"],
   wiring={"phase.reference": "signal_tx"})

y = chain(5_000)
ser = compute_ser(chain.tap("data_tx"), y)
plot_iq(chain.tap("gsop"), title="after GSOP")    # D25: plotting is a function
```

`len(chain.module_list) == 9`, et les neuf sont des blocs de
communication. La même simulation en v0.4 s'écrivait avec deux `Recorder`
et quatre `Scope` intercalés : quinze entrées dans `module_list`, dont six
qui ne traitent rien — et un `Recorder` passé au constructeur du
compensateur, donc une chaîne non sérialisable (D31). Le `wiring` remplace
ce dernier sans réintroduire de référence entre blocs.

**Garde-fous du câblage (c).** La source est tappée automatiquement. Une
arête qui pointe **en arrière** est refusée à l'exécution : elle servirait
la valeur du tirage précédent, c'est-à-dire exactement la panne
silencieuse que D42 supprime. Un identifiant inconnu ou une clé mal
formée échouent en nommant les blocs connus (D38). Le câblage alimente
des *données*, pas de la structure : `__post_init__` n'est pas rejoué —
`set_params` (D34) reste la voie des paramètres qui exigent un
re-précalcul.

**Portée de la sérialisation.** `taps` et `wiring` sont de l'intention au
sens de D31 : ils voyagent avec le document JSON. Une chaîne relue qui
n'enregistrerait plus ce que son auteur a déclaré serait une perte de
fidélité silencieuse, contraire à D32.

**Amendement de D11.** L'ADD v0.3 prévoyait de convertir les *monitors*
en `logging.Logger` configurables. D42 va plus loin et les **supprime** :
un bloc qui journalise reste un bloc parasite dans `module_list`. Ce qui
subsiste de D11 est intact — aucun `print` dans `src/`, le chemin de
débogage passe par `logging` — mais la mesure elle-même est rendue par
`signal_report()`, qui retourne un dictionnaire et laisse l'appelant
décider s'il le journalise, le tabule ou l'assertionne. Séparer la mesure
de sa présentation était le fond du problème.


### 4.12 Modèles de canaux normalisés (nouveau)

| # | Décision | Motif | Alternatives rejetées | Statut |
|---|----------|-------|-----------------------|--------|
| D43 | **Le catalogue de canaux applique à l'axe des retards le motif déjà validé sur l'axe des fréquences.** (a) Un objet valeur gelé `PowerDelayProfile` porte la table (retards en ns, puissances en dB), le spectre Doppler, le facteur de Rice éventuel, **et sa provenance** (`standard`, `reference` = la clause). (b) Registre `get_delay_profile()` / `@register_delay_profile`, jumeau de D17. (c) **Auto-contrôle D20** : l'entrée vérifie à la construction les grandeurs publiées à côté de la table — nombre de trajets, étalement de retard quadratique moyen. (d) La variation temporelle est dans le lot : `rayleigh_process()` synthétise un trajet au spectre de Clarke/Jakes, et `TappedDelayLineChannel` applique le canal, sélectif **en temps comme en fréquence** ; `f_doppler=0` redonne le block-fading. (e) Les grandeurs réalisées sont des attributs soulignés (`h_`, `delays_`, D23) | Toute la machinerie existait déjà — `pdp_to_scales`, `rayleigh_channel`, `SelectiveMIMOChannel` — mais pas le catalogue : il fallait retaper la table du standard à la main, sans filet. C'est exactement la défaillance que D20 a été inventée pour empêcher côté porteuses. Le contrôle n'est pas décoratif : il a immédiatement produit un résultat. EVA et ETU reproduisent leur étalement publié (356,7 ns contre 357 ; 990,9 contre 991) ; **EPA concorde sur deux des trois** : son nombre de trajets (7) et son retard maximal (410 ns) sortent exactement, son étalement non — 43,13 ns contre 45 publiés. L'écart a été instruit, pas toléré : la définition n'est pas en cause (la même formule reproduit EVA à 0,35 ns et ETU à 0,06 ns, et aucune variante — pondération en amplitude 69,8 ns, moment d'ordre deux non centré 61,8 ns, retard moyen 44,2 ns — ne tombe sur 45) ; aucune faute de frappe plausible non plus (il faudrait +1,67 dB sur le dernier trajet, +3,83 dB sur celui à 190 ns, ou +76 ns sur le retard de 410 ns, dans une table en valeurs rondes). Les deux figures concordantes confirment les *retards* ; les *puissances* ne sont confirmées par rien d'indépendant, donc c'est là que se cacherait l'écart. L'entrée épingle les deux chiffres qu'elle reproduit et laisse le troisième en suspens : épingler un chiffre qu'on ne sait pas sourcer serait pire que de n'en épingler aucun. Sur le Doppler, la synthèse se fait **directement sur la grille FFT de sortie** — seuls les bins sous `f_D` sont remplis, une transformée inverse suffit — donc aucun rééchantillonnage ni interpolation, et la réalisation est à bande limitée par construction. La référence de vérification est l'autocorrélation de Bessel `J0(2 pi f_D tau)` | Générateur par somme de sinusoïdes de Jakes (statistiques exactes seulement asymptotiquement, et le choix des phases est un piège classique) ; génération à cadence réduite puis interpolation (introduit une erreur d'interpolation là où la méthode spectrale est exacte) ; renormalisation de chaque réalisation à puissance unité (casse les statistiques gaussiennes : la puissance d'une réalisation *est* aléatoire, c'est physique) | **Acté** |

**Le piège que la décision rend visible.** À 15,36 MHz avec un Doppler de
70 Hz, il faut 219 000 échantillons avant que le canal ne bouge : une
simulation de 4096 échantillons donne du block-fading, silencieusement.
Le générateur le **signale** par `logging` (D11) plutôt que de laisser
prendre un résultat statique pour un résultat sélectif en temps. C'est le
même réflexe que le garde-fou `wdm` de D19.

**Le détail qui décide de la justesse.** Un bin de la grille FFT compte
selon la fraction de bande qu'il **recouvre**, pas selon la position de
son centre. La distinction n'est pas cosmétique : près de `±f_D` la
densité est singulière, donc le bin de bord porte plusieurs pour cent de
la bande. Une première version testait les centres et perdait jusqu'à
**8,3 %** de la puissance selon la longueur de fenêtre — avec un effet de
seuil quand `f_D` tombe pile sur un centre de bin. Le script de validation
balaie quatre grilles précisément pour interdire le retour de ce défaut.

**Limite assumée.** La méthode spectrale est périodique : la réalisation
se répète avec la période `n_samples / fs`. Documenté, pas contourné.

---

### 4.13 Multiplexage en longueur d'onde (nouveau)

| # | Décision | Motif | Alternatives rejetées | Statut |
|---|----------|-------|-----------------------|--------|
| D44 | **Le WDM se décrit par un plan de fréquences, et se réalise par deux blocs symétriques.** (a) Un objet valeur gelé `WDMGrid` porte les fréquences centrales **absolues en hertz**, la bande occupée par canal, le centre de la bande composite et la provenance (`standard`, `reference`) — troisième instance du motif D15/D43, appliqué à l'axe des longueurs d'onde. Constructeurs `WDMGrid.uniform(...)` et `WDMGrid.itu(n_indices, m=…)`, ce dernier bâti sur `itu_grid_frequency` de D19. (b) `WDMMultiplexer` transpose chaque canal sur son créneau et somme : `(..., C, N) -> (..., N)`, canaux sur l'axe −2, la convention d'antennes de D2. (c) `WDMDemultiplexer` fait l'inverse — transposition retour puis masque rectangulaire sur la DFT — et rend `(..., C, N)`, ou `(..., N)` avec `channel=c` pour un récepteur accordé sur une seule longueur d'onde. (d) **Les fréquences sont absolues et en hertz, et tout bloc qui en a besoin prend un `fs` explicite** : c'est la convention des bibliothèques de radio logicielle (`freq_xlating_fir_filter`, `pfb_synthesizer_ccf` / `pfb_channelizer_ccf` de GNU Radio), et le multiplexeur *est* le banc de synthèse, le démultiplexeur le banc d'analyse. (e) **Aucun changement de cadence dans ces blocs** : ils entrent et sortent à la cadence composite ; le sur-échantillonnage et la décimation restent `Upsampler`/`Downsampler`. (f) La grille se refuse à la construction si la bande occupée dépasse l'espacement, et `validate_fs` refuse une cadence qui replierait le peigne | Jusqu'ici la lib **rejetait** le WDM sans l'offrir : `FiberLink.prepare` levait une `ShapeError` disant « multiplexez d'abord », sans fournir de quoi le faire — un message qui nomme une issue inexistante est pire qu'un message générique (D38 exige constat/attendu/**action**). Le plan de fréquences était présent (`itu_grid_frequency`, D19) mais rien ne le consommait. Séparer le plan des blocs est ce qui rend la grille partageable entre émetteur et récepteur, sérialisable (D31, elle est dans `_VALUE_MODULES`) et affichable — son `__repr__` dessine la carte spectrale ASCII de D21. L'axe −2 pour les canaux n'est pas un choix libre : c'est déjà celui des antennes MIMO, et un WDM à `(C, N)` se lit alors comme un MIMO, ce qui est exactement le bon parallèle. Le refus de rééchantillonner est ce qui garde un bloc = un rôle : un multiplexeur qui interpolerait cacherait un filtre de mise en forme dans un bloc dont le nom ne le dit pas. **Le point de justesse** : le masque est appliqué sur la DFT du bloc, donc l'opération est *circulaire* — le décalage n'est exact que si chaque `f_c` tombe sur un bin entier (`f_c N / f_s` entier). Quand ce n'est pas le cas le bloc le **signale** par `logging` (D11) : mesuré, l'aller-retour passe de 1,8e−13 (grille alignée) à 17 % d'erreur relative et 5 % de diaphonie sur une longueur de bloc quelconque. Se taire ici aurait produit un plancher de diaphonie que l'utilisateur aurait attribué à la fibre | Fréquences normalisées à `fs` (ce que fait `scipy.signal` ; mais un plan WDM est physique et publié en THz — normaliser oblige à reconvertir à chaque lecture de standard, et rend le même objet faux dès qu'on change `fs`) ; longueurs d'onde en nm plutôt qu'en hertz (l'espacement est constant en fréquence, pas en longueur d'onde : la grille ITU est définie en THz) ; banc de filtres polyphase à la GNU Radio (le gain de complexité est réel en temps réel, nul ici où tout est hors ligne, et il ajoute un filtre prototype à choisir — donc un paramètre de plus qui change le résultat) ; multiplexage avec changement de cadence intégré (viole un bloc = un rôle, et cache un filtre d'interpolation non nommé) ; masque à flancs adoucis par défaut (introduit un paramètre de roll-off que rien dans le modèle ne fixe ; le mur rectangulaire rend l'aller-trip exactement inversible, ce qui est *testable*) | **Acté** |

**Ce que la décision rend vérifiable.** L'aller-retour est un test binaire,
pas une tolérance négociée : sur une grille alignée sur les bins, cinq
canaux à 50 GHz d'espacement et 32 GHz de bande se reconstruisent à
1,8e−13 en erreur relative, la diaphonie est à 1,6e−13, et la puissance
du peigne est exactement la somme des puissances des canaux — les
créneaux disjoints sont orthogonaux. Un défaut d'axe, de signe ou de
normalisation casse l'un des trois.

---

### 4.14 Amplification Raman distribuée (nouveau)

| # | Décision | Motif | Alternatives rejetées | Statut |
|---|----------|-------|-----------------------|--------|
| D45 | **Le Raman se découpe en trois objets, selon ce dont chaque grandeur est une propriété.** (a) `RamanGainSpectrum` porte la **forme** normalisée du gain contre le décalage Stokes, gelée, avec `standard`/`reference`, registre et **auto-contrôle D20** — quatrième instance du motif D15/D43, appliqué à l'axe des décalages Raman. Deux paramétrages mutuellement exclusifs à la D41 (`lorentzian=(tau1, tau2)` ou `triangular=peak_THz`), aucun argument discriminant. (b) Le coefficient crête `g_R/A_eff` **n'y est pas** : c'est une propriété de la *fibre*, pas du verre — SMF, DCF et NZDSF diffèrent d'un grand facteur par l'aire effective et le dopage — donc c'est un argument du solveur, à côté des pertes. (c) `solve_raman()` intègre les équations couplées de puissance : **problème à valeur initiale** (`solve_ivp`) quand seule la pompe co-propagative est allumée, **problème aux limites** (`solve_bvp`) sinon, avec les profils non dépletés comme germe. (d) **Aucun `Processor`** : le Raman vit dans le domaine des puissances, le SSFM dans celui du champ ; ce qui sort est le profil `G(z)`, destiné au pas linéaire de `FiberLink`. (e) La direction n'est **pas** un paramètre : `pump_forward_W` / `pump_backward_W` — quelles pompes sont allumées *est* la configuration | Un bloc appliquant un gain forfaitaire en fin de span décrirait un amplificateur **discret**, c'est-à-dire la seule chose que l'amplification distribuée n'est pas : tout son intérêt est **où** le gain a lieu, ce qui change l'accumulation de bruit et la pénalité non linéaire, pas seulement la puissance de sortie. La mesure le montre — à 500 mW le co-pompage a délivré plus de 70 % de son gain à mi-span, le contra-pompage moins de 30 %. **Ce qui rend le module acceptable sous D7, c'est qu'il se vérifie sur cinq références dont trois couvrent le régime dépleté.** La plus forte est la **solution exacte du cas simple** : sans pertes, la conservation du nombre de photons élimine la pompe et il reste une **équation logistique**, dont la solution fermée vaut sous déplétion *arbitraire* — et avec des pertes **égales** la même solution tient en longueur effective, la substitution `Q = P e^{alpha z}` ramenant le couple au cas sans pertes. Elle épingle **tout le profil**, pas un chiffre de sortie : écart mesuré **4,0e−11** sans pertes et **3,4e−11** à pertes égales, avec la pompe consommée à 100 %. La forme fermée non dépletée `exp(g P_p L_eff)`, elle, est exacte mais muette dès que le signal mange la pompe : la forme fermée non dépletée `exp(g P_p L_eff)` est exacte mais muette dès que le signal mange la pompe, or c'est précisément là qu'un solveur numérique peut être faux et paraître juste. La **conservation du nombre de photons** `P_s/nu_s + P_p/nu_p` en limite sans pertes tient sous déplétion arbitraire : mesurée à **2,9e−15** avec la pompe dépletée de plus de 20 %, et c'est elle qui attrape un facteur `nu_p/nu_s` erroné, que le contrôle non déplété ne voit pas du tout. S'ajoutent la convergence des trois schémas vers la forme fermée à faible pompe (0,0007 à 0,0069 dB) et l'écart contra−co qui s'ouvre **monotonement** de 0,0061 dB à 50 mW jusqu'à 2,93 dB à 1 W — non trivial, et une erreur de signe sur le retournement de direction le casse. Sur les modèles de spectre : Blow–Wood place le pic à **13,08 THz** contre 13,2 publiés (1 %), ce que l'auto-contrôle épingle, mais donne une largeur à mi-hauteur de **9,55 THz** là où la silice mesurée fait 5 à 6 — 70 % trop large. C'est écrit, pas caché, et un test l'épingle pour que personne ne « corrige » la largeur en bougeant les constantes de temps, ce qui casserait le pic que la source, elle, spécifie. Le fit multi-lorentzien qui reproduirait la forme **n'est pas livré** : ses coefficients n'ont pas été transcrits depuis leur source, et les inventer serait exactement la faute que P3 interdit | Un `Processor` Raman appliquant un gain forfaitaire (décrit un ampli discret, faux pour du distribué, et masquerait que le profil `G(z)` est le vrai livrable) ; un paramètre `direction="co"/"counter"` (redondant avec les puissances de pompe, donc contradictoire dès qu'on se trompe — même faute que le couple `value`/`unit` de D41) ; `solve_bvp` uniformément, y compris en co-pompage (le cas co est un IVP exact qui ne peut pas ne pas converger ; le résoudre en BVP ajoute un risque pour rien — les deux chemins sont comparés dans le script de validation, à 1,4e−6 dB) ; germe plat pour le BVP (diverge dès que la déplétion est notable ; le profil non déplété converge jusqu'à 1 W) ; retourner un maillage non convergé (il ressemble exactement à un résultat plausible — `status != 0` lève, message D38) ; mettre `g_R` crête dans le spectre (fige une fibre dans un objet qui décrit le verre) | **Acté** |
| D45b | **Le solveur décrit un ensemble d'ondes, pas une paire.** Chaque canal et chaque pompe, dans chaque direction, est une onde `P_i(z)` de direction `d_i = ±1` ; **toutes les paires** sont couplées par `C_ij` construite au décalage qui les sépare, avec `C_ji = -(nu_j/nu_i) C_ij`. Tout argument décrivant un signal ou une pompe accepte un scalaire — partagé par le groupe — ou une valeur par onde, et **un scalaire en entrée redonne un scalaire en sortie**. `spectrum=` devient obligatoire dès qu'un groupe compte plus d'une longueur d'onde. Côté champ, `FiberLink` transforme le gain multi-canal en **fonction de transfert** interpolée sur la grille FFT, appliquée aux demi-pas comme le gain plat | Le multi-pompe et le tilt inter-canaux ne sont pas des extensions du modèle : ils **sont** le même modèle, dès lors qu'on cesse de privilégier une paire. Le gain pompe→signal, le transfert pompe→pompe (donc le pompage d'ordre deux) et le tilt du peigne tombent d'un seul jeu d'équations, sans une ligne de code par effet. **Deux références analytiques nouvelles portent précisément les deux axes ajoutés** : la forme fermée de Zirngibl (1998) reproduite à 0,7 % du tilt — la taille de l'approximation qu'elle fait — et la somme des gains non dépletés multi-pompes, dont le résidu se divise par **101** quand la pompe se divise par dix, ce qui l'identifie comme le transfert pompe→pompe et non comme une erreur. L'EDFA restant plat, il compense le gain **moyen** : les canaux sortent répartis autour de la transparence sur la largeur du tilt, ce qui est la physique, pas un raccourci | Une classe séparée `MultiPumpRaman` (le même système d'équations, dupliqué) ; garder `spectrum=None` en multi-onde (signifie « chaque paire au pic », donc un tilt inventé et silencieux) ; appliquer le gain multi-canal par ligne sur un axe de canaux (la fibre voit **un** champ multiplexé, pas des canaux séparés — ce serait décrire des fibres parallèles) ; extrapoler le tilt au-delà du peigne résolu (inventer du gain là où rien n'a été résolu ; le bord est tenu) | **Acté** |

**Seconde passe : le crochet dans `FiberLink`.** `FiberLink(...,
raman=solution)` échantillonne le profil aux bornes de chaque pas SSFM
et applique le gain **dans la boucle**, de sorte que le terme de Kerr
voie la puissance que la fibre porte réellement — c'est toute la raison
d'utiliser un profil plutôt qu'un gain forfaitaire. Trois conséquences
assumées : (i) l'EDFA de fin de span est **réduit du gain on-off**, donc
un span reste transparent qu'il soit pompé ou non (sans quoi la
puissance croîtrait de span en span) ; (ii) l'ASE que le solveur a
intégrée jusqu'à `z = L` est ajoutée une fois par span, ramenée de sa
bande de référence à `fs` ; (iii) le lien reçoit une graine, qui
alimente l'ASE Raman **et** l'EDFA — celui-ci se construisait sans
graine, donc `FiberLink` bruité n'était pas reproductible, et semer la
moitié des sources aurait été pire que n'en semer aucune. Les deux
propriétés qui rendent l'intégration vérifiable : un span pompé sort à
la puissance où il est entré (à 1e−6 près, dans les deux modes de
propagation), et un gain Raman évanescent reproduit le lien non pompé à
**1e−12** — c'est le test de non-régression des chaînes existantes. Un
défaut trouvé en écrivant ces tests : la branche `use_only_linear` n'a
pas de boucle de pas, donc le gain n'y était pas appliqué alors que
l'EDFA était réduit — **15 dB perdus en silence**.

**Le raccord des deux grilles, qui est le vrai point technique.** Le
solveur Raman et le SSFM ont des maillages **indépendants** : le
solveur ne connaît pas `StPS`, le SSFM ne connaît pas `n_nodes`. Le
raccord se fait en interpolant le gain **cumulé** `G(z)` aux bornes des
pas et en le différenciant. Deux conséquences, mesurées :

* le gain **total est exact quel que soit `StPS`** — la somme des
  incréments télescope en `G(L) − G(0)` — donc un span reste
  transparent à 1e−9 de `StPS = 1` à `StPS = 400` ;
* seule la **répartition** du gain est interpolée, et l'erreur qu'elle
  laisse porte sur la phase non linéaire, pas sur la puissance.

**Le gain appartient à l'opérateur linéaire, donc il se scinde comme
lui.** Appliqué une fois par pas, il casse la symétrie du split-step
symétrique et fait **tomber le schéma du second au premier ordre** :
mesuré sur la phase SPM d'une onde continue, l'erreur décroissait en
1/StPS au lieu de 1/StPS². Le profil est donc échantillonné aux
**demi-pas** et les deux incréments encadrent le terme de Kerr, comme
les deux demi-pas de dispersion. Ordre rétabli (rapport ~4 par
doublement) et erreur divisée par **24** à `StPS = 20`.

**Deux réglages que la mesure fixe.** (i) `n_nodes = 401` par défaut sur
le solveur : le *gain* est insensible au maillage dès 21 nœuds, mais la
*forme* du profil ne l'est pas — l'écart de phase tombe de 2,3e−4 rad à
21 nœuds à 6e−12 à 401. (ii) Les pas **logarithmiques** sont
contre-indiqués avec du Raman, et le bloc le signale par `logging` : ils
sont dimensionnés pour la décroissance exponentielle d'un span non
pompé, que l'amplification distribuée aplatit — les pas les plus longs
atterrissent alors là où la puissance est la plus forte. Mesuré, la
grille logarithmique est **3× pire** que la linéaire dès que le Raman
est allumé (8,2e−4 rad contre 2,8e−4 à `StPS = 20`), alors qu'elle est
légèrement meilleure sans.

**Troisième passe : multi-pompe et multi-signal (D45b).** Le solveur ne
décrit plus une paire pompe-signal mais un **ensemble d'ondes**. Chaque
onde — canal, pompe, dans chaque direction — est une puissance
`P_i(z)` écrite dans la coordonnée `+z` avec une direction
`d_i = ±1`, et **toutes les paires sont couplées** par une matrice
`C_ij` construite à partir du spectre au décalage qui sépare les deux
ondes, avec `C_ji = -(nu_j/nu_i) C_ij`. Trois effets tombent alors du
même jeu d'équations, sans code dédié : le gain pompe→signal, le
transfert **pompe→pompe** (donc le pompage d'ordre deux) et le **tilt
inter-canaux** — le peigne se pompe lui-même, ses canaux bleus
alimentant ses rouges. `spectrum=` devient **obligatoire** dès qu'il y a
plus d'une longueur d'onde dans un groupe : sa valeur par défaut
signifie « la paire est au pic du gain », défendable pour une paire,
absurde pour un peigne. Un scalaire reste partagé par tout un groupe,
et **un scalaire en entrée redonne un scalaire en sortie** : le cas
mono-canal lit exactement comme avant.

**Deux références analytiques nouvelles, qui portent précisément les
deux axes ajoutés.** (i) Sur l'axe multi-signal, la **forme fermée de
Zirngibl** (Electron. Lett. 1998) : avec un gain triangulaire et
`nu_i/nu_j ≃ 1`, les termes communs se factorisent et il reste une
repondération des canaux en `exp(-C_R nu_i P_tot L_eff)`. Le solveur la
reproduit à **5,1e−4 dB** pour un tilt de 0,078 dB, **1,1e−2 dB** pour
un tilt de 1,56 dB — c'est-à-dire à 0,7 % du tilt, exactement la taille
de l'approximation que le modèle publié fait, et pas celle du tilt.
(ii) Sur l'axe multi-pompe, la **somme des gains non dépletés** : le
solveur s'en écarte de 1,37e−2 dB (0,77 % du gain), et diviser la
puissance de pompe par dix divise cet écart par **101**. Cet exposant
est le contrôle : un résidu quadratique *est* le transfert
pompe→pompe, un résidu linéaire aurait été une erreur. S'y ajoute la
conservation du nombre de photons sur **dix ondes** simultanées
(1,4e−15 avec les pompes dépletées à 100 %), qu'une matrice de couplage
ayant perdu le facteur `nu_j/nu_i` sur *une seule* paire ne passerait
pas.

**Le raccord au champ : le gain devient un filtre.** Le multiplexeur
D44 somme les canaux en **un seul champ** avant la fibre, donc un gain
par canal n'est pas une multiplication par ligne : c'est une **fonction
de transfert** sur la bande simulée. `FiberLink` interpole les canaux
résolus sur la grille FFT — bord tenu, jamais extrapolé, extrapoler un
tilt Raman hors du peigne revient à inventer du gain — et l'applique
demi-pas par demi-pas, là où le gain plat allait. **Ce que cela rend
visible et qui est vrai** : un EDFA est plat, il ne peut pas défaire un
tilt ; il compense donc le gain **moyen** et les canaux sortent
répartis autour de la transparence sur la largeur du tilt. C'est la
situation physique qu'un égaliseur de gain existe pour corriger, pas un
raccourci de modélisation, et `raman.tilt_dB` la chiffre.

**Ce qui reste hors périmètre.** Le fit multi-lorentzien de la silice
n'est toujours pas livré — ses coefficients n'ont pas été transcrits
depuis une source, et les inventer serait la faute que P3 interdit ; le
modèle Blow–Wood reste 70 % trop large en largeur à mi-hauteur, ce qui
est écrit dans sa docstring. L'ASE d'une solution multi-canal est
ajoutée **plate** sur la bande, moyennée sur les canaux, alors que le
gain, lui, est mis en forme : c'est dit à l'endroit où c'est fait.

**Un résultat que la mesure a tranché.** Le co-pompage donne le
**meilleur** facteur de bruit — 6,78 dB contre 14,83 à 500 mW — parce
que le gain est délivré tant que le signal est encore fort. Le
contra-pompage reste préféré en pratique pour une raison que ce modèle
de puissances ne porte pas : le transfert de RIN de la pompe. C'est dit
dans la docstring plutôt que laissé à l'intuition du lecteur.

---

### 4.15 Fibre et simulation, séparées (nouveau)

| # | Décision | Motif | Alternatives rejetées | Statut |
|---|----------|-------|-----------------------|--------|
| D46 | **Ce qu'est la fibre se sépare de la façon dont on la simule.** (a) `FiberSpec` gelé porte les coefficients physiques — perte, Kerr, dispersion, longueur d'onde, gain Raman crête — **et leur provenance** ; quatrième instance du motif D15/D43/D45, avec registre (`get_fiber`, `@register_fiber`) et auto-contrôle D20. (b) `FiberLink` et `DBP` prennent `fiber=FiberSpec(...)` ; il leur reste les paramètres de simulation (`StPS`, `step_type`, `fs`, `L_span`, …). `FiberLink` passe de **21 à 15** arguments constructibles. (c) La fréquence porteuse est **dérivée** de la longueur d'onde, plus un paramètre. (d) `c` et `h` disparaissent de la surface : ce sont des constantes universelles, pas des réglages. (e) **Garde-fous d'unité** : chaque grandeur physique est bornée par la *largeur d'une faute d'unité*, et le message nomme l'unité attendue | Les deux familles changent pour des raisons sans rapport : on remplace une SMF par une DCF sans toucher au pas d'intégration, et on raffine `StPS` sans toucher au verre. Les mélanger obligeait à retaper quatre nombres physiques au milieu de réglages de solveur, sans rien qui les relie ni qui dise d'où ils viennent. **L'incohérence que la séparation supprime** : `lamb` et `nu` étaient deux arguments indépendants qui doivent s'accorder — une chaîne réglée à 1310 nm calculait sa dispersion à 1310 et l'énergie photonique de son ASE à 1550, les deux ne s'accordant que parce que les deux valeurs par défaut s'accordaient. C'est exactement le défaut que D41 interdit. **Sur les unités, le motif a payé immédiatement** : en écrivant `FiberSpec.beta2` j'ai recopié la conversion `D -> beta2` au lieu de déléguer à `compute_beta2`, avec un facteur **1000** d'erreur — et l'auto-contrôle D20 de l'entrée SMF l'a rejeté à la construction, en nommant l'écart (−0,0217 contre −21,7 attendu). Le correctif structurant est la délégation, pas la borne : une grandeur calculable de deux façons doit l'être d'une seule. Les bornes de plausibilité couvrent le cas que le catalogue ne voit pas, les valeurs que l'utilisateur tape ; elles sont **délibérément lâches** — ce sont la taille d'une confusion dB/m contre dB/km, pas des lois de la physique, et un refus y est une unité, pas une question de recherche | Deux classes `FiberLink`/`RamanFiberLink` (découpe selon le mauvais axe : le Raman ajoute 2 arguments sur 21, pas la cause de l'encombrement) ; garder `c` et `h` réglables (aucune raison légitime de redéfinir une constante universelle, et ça fait deux paramètres qui peuvent contredire le reste) ; bornes serrées calées sur les fibres du catalogue (rejetterait une fibre exotique légitime — un garde-fou qui bloque du travail valide est pire que pas de garde-fou) ; `nu` gardé en argument avec validation croisée contre `lamb` (deux sources pour une grandeur, alors que la dériver rend le désaccord impossible) | **Acté** |

**Ce que la CI a attrapé et que la machine locale ne pouvait pas.** Un
doctest de `DataAidedFIRCompensator` passait en Python 3.11 et échouait
en 3.12 : `np.round` produisait `-0.` au lieu de `0.`. IEEE 754 a deux
zéros, `numpy` les affiche différemment, et lequel sort dépend du
dernier bit avant arrondi — donc de la version, de la plateforme et des
options de compilation. Sept autres doctests affichaient un zéro arrondi
et étaient à une version de `numpy` du même sort. Règle ajoutée et
outillée : **aucune sortie attendue de doctest ne contient de zéro
négatif**, et on l'évite en ajoutant `+ 0.0` au tableau arrondi, IEEE
garantissant `-0.0 + 0.0 = +0.0`.

**Un défaut préexistant révélé au passage.** `FiberLink` n'a **jamais**
été sérialisable : son champ `callbacks` vaut `{}` par défaut, et le
codeur refusait tout dictionnaire — alors que D31 exige que tout bloc
fasse l'aller-retour. Le dictionnaire *vide* passe désormais ; un
dictionnaire peuplé reste refusé, il contient des callables, la
frontière assumée.

---

## 5. API d'allocation (résumé normatif)

```python
class CarrierType(IntEnum):
    NULL = 0; DATA = 1; PILOT = 2

@dataclass(frozen=True)
class CarrierAllocation:
    carrier_type: np.ndarray            # (T_p, N_fft) int8, physical order
    subcarrier_spacing: float | None = None
    cp_length: int | None = None
    standard: str = "custom"
    reference: str = ""                 # clause of the standard
    # invariants: ndim == 2; N_data constant over the period
    # properties: N_fft, N_data, N_pilots, period, k, summary
    # methods:    to_fft_order(), plot(), __repr__()

def band_allocation(N_fft, k_used, k_pilots=(), n_dc=1, expect=None, **meta): ...
def scattered_allocation(N_fft, k_used, period, rule, expect=None, **meta): ...
def get_allocation(standard: str, **kwargs) -> CarrierAllocation: ...
```

Une entrée de catalogue se lit comme le tableau du standard dont elle
sort, et se contrôle elle-même :

```python
@register_allocation("802.11a")
def _wifi_11a():
    return band_allocation(
        N_fft=64,
        k_used=(-26, 26),                    # 52 occupied subcarriers
        k_pilots=(-21, -7, 7, 21),
        n_dc=1,
        expect={"data": 48, "pilots": 4},    # <- Table 17-5, copied verbatim
        subcarrier_spacing=312.5e3, cp_length=16,
        standard="802.11a", reference="IEEE 802.11-2020, Table 17-5",
    )
```

**Rendu (D21b).** Glyphes : `#` data, `P` pilote, `.` nulle.

```
>>> print(get_allocation("802.11a"))
802.11a (64-FFT, 20 MHz)   [IEEE 802.11-2020 Table 17-5]
    |                               0                              |     k=-32..+31
    ......#####P#############P######.######P#############P#####.....
    data 48 | pilotes 4 | nulles 12 | 312.5 kHz, CP 16
```

Deux régimes d'affichage, et la règle n'est pas cosmétique :

- `T_p == 1` — vue pleine bande ; si `N_fft` dépasse la largeur du
  terminal, agrégation par blocs avec priorité `pilote > data > null`.
- `T_p > 1` — l'agrégation est **interdite** : sur un motif dispersé
  chaque bloc agrégé contient un pilote et la carte devient une bande de
  `P` uniforme, c'est-à-dire rien. On affiche un zoom pleine résolution
  sur quelques périodes, ce qui restitue la structure en diagonale :

```
>>> print(get_allocation("DVB-T", mode="2K"))
l=0 P###########P###########P###########P###########P###########
l=1 ###P###########P###########P###########P###########P########
l=2 ######P###########P###########P###########P###########P#####
l=3 #########P###########P###########P###########P###########P##
    data 1562 | pilotes 143 | nulles 343
```

**Catalogue initial.** Familles retenues par usage réel : Wi-Fi
(enseignement, prototypage), LTE/NR (référence cellulaire), DVB-T
(démonstrateur de motif périodique).

| Clé | `N_fft` | `k_used` | Pilotes (`k`) | DC nuls | data / pilotes |
|---|---|---|---|---|---|
| `802.11a` (= 11g, 20 MHz) | 64 | ±26 | ±7, ±21 | 1 | 48 / 4 |
| `802.11n` (HT, 20 MHz) | 64 | ±28 | ±7, ±21 | 1 | 52 / 4 |
| `802.11ac` (40 MHz) | 128 | ±58 | ±11, ±25, ±53 | 3 | 108 / 6 |
| `802.11ac` (80 MHz) | 256 | ±122 | ±11, ±39, ±75, ±103 | 3 | 234 / 8 |
| `LTE` (`bandwidth_MHz`) | 128…2048 | ±(6·N_RB) | — (CRS séparé) | 1 | 12·N_RB |
| `5G-NR` (`mu`, `N_RB`) | ≥ 12·N_RB | −6·N_RB … 6·N_RB−1 | — (DM-RS séparé) | 0 | 12·N_RB |

LTE : espacement 15 kHz, bloc de ressources = 12 sous-porteuses,
`N_RB` ∈ {6, 15, 25, 50, 75, 100} pour 1,4 / 3 / 5 / 10 / 15 / 20 MHz.
5G NR : espacement `15 × 2^μ` kHz, `μ ∈ {0..4}`, bloc = 12 sous-porteuses ;
contrairement à LTE, NR ne réserve pas de sous-porteuse DC.

> ⚠️ **Ces valeurs sont à revalider ligne à ligne contre le texte des
> spécifications avant d'être fusionnées.** Elles ont été contrôlées pour
> cohérence interne uniquement (les sommes tombent juste : 48+4+1+11 = 64
> pour 802.11a). D20 en fait une condition de merge.

---

## 6. Structure cible

```
src/comnumpy/
  _backend.py  (nouveau — D3, aiguillage FFT interne)
  sweep.py     (nouveau — D35)
  core/        (mise en conformité D2/D10/D11/D23, LLR D12 ; trames D28–D29 ;
                monitors.py supprimé, visualizers.py en fonctions — D42)
  ofdm/        (S/P en reshape pur ; allocation D15–D21)
  mimo/        (déjà conforme (ant, N) ; einsum '...ij,...jt->...it')
  optical/     (golden tests D7 prioritaires ; grille WDM D19)
  fec/         (nouveau — D4 puis D5)
validation/    (nouveau — D7)
docs/          (+ page CONVENTIONS)
ARCHITECTURE.md (ce document), CONVENTIONS.md, CITATION.cff, CHANGELOG.md
```

---

## 7. Ordre d'exécution

**Lot 0 — assainissement** *(≈ 1 jour, avant tout le reste)*
D8 (README + doctest global en CI) · D1 (retrait `seaborn`/`tqdm`,
README aligné sur les dépendances réelles) · D9 (décision de version) ·
**D39** (ruff + pre-commit + cliquet de couverture : à poser *avant* le
refactor D2, pour que toute la suite passe dessus) ·
correction des défauts bloquants de l'annexe A · CI sur les trois
versions de Python annoncées.

1. **D2 + D36 + D38 + D40a/b + D41** — refactor axes, suppression de
   `is_mimo`, `prepare()` validants, page CONVENTIONS ; **même fenêtre de
   casse** : surface publique `__all__`, hiérarchie d'exceptions,
   `slots`/`kw_only`, kwargs de paramétrage (`AWGN(snr_dB=…)`). Ces
   cinq-là cassent des signatures — les grouper évite des breaking
   changes successifs *(1,5 j)*
2. **D7** — `validation/`, **optique d'abord** *(2 w-e)*
3. **D15–D21** — allocation spectrale et grille WDM *(1 w-e)*
4. **D22–D24, D28–D29, D34** — conventions d'estimateurs, structure de
   trame, uniformisation des compensateurs, `set_params` (D34 dépend de
   la restriction sur `__init__` posée au même jalon, et les scripts
   `validation/` du jalon 2 en ont besoin dès leur deuxième figure) *(1 w-e)*
5. **D4/D12** — `fec/` + LLR demapper + golden BER *(1 w-e)*
5bis. **D33 puis D31/D32** — `__repr__`, `summary()`, `to_mermaid()`, puis
   export JSON et test d'aller-retour. Peu coûteux et immédiatement utile :
   `summary()` sert le refactor D2, le JSON sert D6 et le papier *(2 après-midi)*
5ter. **D35** — *si et seulement si* le déclencheur est atteint (3ᵉ script
   `validation/` au squelette identique) : extraction de `sweep()` depuis
   ces scripts *(1 après-midi)* ; sinon, reporté post-1.0
6. **D6, D10, D11, D23, D25, D27, D30, D37** — graine de chaîne,
   docstrings au gabarit §4.10 (validation `numpydoc` en CI dès le
   premier fichier converti), logging, gabarit de tracé, palette,
   feuille de style, passe d'anglicisation, annotations de type au fil
   des fichiers touchés *(au fil de l'eau ; **porte stricte à la
   v1.0** : `pyright` strict vert et `py.typed` livré, sinon pas de tag)*
6bis. **D5, D3** — LDPC min-sum et `_backend.py`. *Remontés v0.5* :
   initialement prévus « versions suivantes », les deux se sont révélés
   peu coûteux une fois D4 en place (même philosophie de vectorisation)
   et sans effet sur l'API publique *(1 après-midi chacun)*
6ter. **D42** — suppression des blocs d'instrumentation, `taps` et
   `wiring`. Non planifié en v0.4 : le besoin est apparu à l'usage, la
   chaîne ne décrivant plus le système. Le faire **avant** le tag v1.0
   est impératif — c'est une rupture d'API, et la fenêtre de casse se
   ferme au jalon 7 *(1 j, migration des exemples et de la doc comprise)*
7. **D9** — tag `v1.0.0`, CITATION, CHANGELOG
8. Papier JOSS *(2 w-e ; les figures sortent de `validation/`)*

---

## 8. Risques et points ouverts

**`optical/` reste la zone la plus risquée.** L'audit le confirme
quantitativement : 0 test sur le module le plus cité, contenant un bloc
mort. D7 y est prioritaire — une erreur de normalisation SSFM publiée
dans une courbe serait le pire scénario.

**Axe `wdm` et non-linéarité (D19) — piège actif.** `FiberLink.forward`
applique le Kerr de façon ponctuelle sur tout le tableau, ce qui n'est
correct qu'en simulation *full-field* : un tableau unique échantillonné
sur toute la largeur du peigne, où SPM, XPM et FWM émergent de `|E|²E`.
Si un tableau portant un axe `wdm` atteint ce code, il produira
silencieusement de la SPM seule — pas de XPM, pas de FWM — donc des
courbes fausses et optimistes, sans lever d'erreur. `prepare()` doit
refuser :

```
ValueError: nonlinear propagation requires a full-field signal (..., N);
got an array with a WDM axis. Multiplex the channels with WDMMultiplexer
first, or use a coupled-NLSE model (not implemented).
```

**Breaking change D2 + D9.** Assumé, documenté dans le CHANGELOG avec
table de migration (anciennes formes → nouvelles). Pas de couche de
compatibilité. `get_standard_carrier_allocation()` est conservée un temps
comme enveloppe mince marquée `DeprecationWarning`.

**Performance asymétrique.** Le détecteur ML MIMO boucle en Python sur
les symboles (~15 s par point de SNR). Livrer un Viterbi vectorisé batch
(D4) pendant que la détection MIMO reste scalaire fragilise le message
« vectorisé ». À inscrire comme item explicite d'une version ultérieure,
ou à corriger.

**Points ouverts :**

- *Poinçonnage FEC* (rates 2/3, 3/4) : masque sur la sortie de
  l'encodeur ; à spécifier quand un usage réel le demande.
- *Métriques de branche Viterbi* : le prototype construit `bm` en une
  fois sur `(batch, T, S, 2)` (~106 Mo pour 200 trames × 518 pas × 64
  états). À calculer dans la boucle temporelle ou par chunks de trames.
- *Gabarit des messages `prepare()`* : uniformiser sur le modèle
  « expected shape (..., ant, N), got (N,) — add an antenna axis or
  use... ».
- *Séquence de polarité des pilotes 802.11* (pseudo-aléatoire, longueur
  127) : relève de l'argument `pilots`, pas du masque. Utilitaire
  `wifi_pilot_polarity(T)` à fournir quand un usage le demandera.
- *Signaux de référence 2D* (CRS LTE, DM-RS NR) : positions exprimables
  par `scattered_allocation`, mais leurs valeurs dérivent de séquences de
  Gold / Zadoff-Chu. À traiter avec l'estimation de canal.
- *Unités d'allocation* : faut-il un `sample_rate` dans
  `CarrierAllocation` pour rendre `subcarrier_spacing` (Hz) et
  `cp_length` (échantillons) convertibles, ou reste-t-il une propriété de
  la chaîne ?
- *Super-canaux WDM* : groupe de `WDMChannel` adjacents traité comme une
  entité logique. À spécifier si le module optique en a l'usage.
- *Trames de longueur variable* — **limite de périmètre à assumer**. Un
  tableau numpy est de forme fixe : `FrameStructure` décrit donc des
  trames de longueur constante, diffusées sur les axes de tête. Le cas
  paquet réel (longueur portée par un en-tête, trames hétérogènes dans
  un même flux) demande le mécanisme de *tagged streams* de GNU Radio,
  qui est un autre modèle d'exécution. Hors périmètre v1.0 ; à écrire
  explicitement dans la doc plutôt qu'à laisser découvrir.
- *Graphe de calcul* — **frontière de périmètre.** `Sequential.forward`
  est une boucle linéaire (`for p in module_list: Y = p(Y)`), donc une
  seule entrée et une seule sortie. Les besoins réels de branche
  (traitement par antenne puis recombinaison), d'entrées multiples
  (égaliseur ayant besoin du signal *et* de l'estimée) et de rebouclage
  (décision-dirigé, turbo) n'y entrent pas. La preuve que ça mordait
  déjà : `DataAidedComplexGainCompensator` (alors `TrainedBased…`)
  recevait `extractor` et sa référence **par le constructeur**, faute de pouvoir les recevoir
  par la structure — des arêtes de graphe qui passaient par la
  configuration. *Traité partiellement v0.5* : **D42c** couvre ce cas
  précis, l'entrée supplémentaire d'un estimateur, en la déclarant sur la
  chaîne (`wiring`) plutôt que dans le constructeur du bloc. Restent hors
  périmètre la branche et la recombinaison. Trois niveaux, par coût
  croissant : liste (état actuel, **plus une arête de donnée déclarée,
  D42c**), DAG (branches et entrées multiples), graphe cyclique
  (rebouclage, autre modèle d'exécution — c'est le métier de GNU Radio).
  D31 garde la porte ouverte au niveau 2 sans l'implémenter ; le niveau 3
  est explicitement hors périmètre v1.0
- *Champ d'en-tête (SIG) et CRC* : relèvent du même objet
  `FrameStructure`, mais introduisent un décodage conditionnel (la
  longueur dépend de l'en-tête). À spécifier avec le point précédent.
- *Format de scénario sérialisé* — extension naturelle de D31 + D35 : un
  fichier décrivant chaîne (JSON D31), balayage (`param`, `values`),
  graine (D6) et métriques, exécutable par `sweep()` sans code. Utile
  pour le BANC (rejouer une campagne de mesures) et comme argument de
  reproductibilité JOSS, mais c'est un format de fichier, pas une API :
  à spécifier après que D35 existe, jamais avant. Danger identifié à ne
  pas franchir : le patron *Trainer* à hooks (Lightning), rejeté en D35,
  ne doit pas revenir par ce chemin.
- *Contenu exact de la surface publique (D36)* : la liste nominative de
  ce qui s'exporte depuis `comnumpy` (racine) versus les sous-paquets
  n'est pas encore arrêtée. À figer en écrivant les `__all__` du jalon 1,
  avec une règle de départ : la racine expose ce dont l'exemple canonique
  D40c a besoin, plus `Sequential`/`Processor` ; le reste vit dans son
  sous-paquet. Toute promotion ultérieure vers la racine est additive,
  donc non cassante — dans le doute, exporter bas.
- *Attribution des `id` de blocs (D31/D34)* : la notation pointée de
  `set_params` suppose des identifiants adressables avant toute
  sérialisation. Règle proposée, à confirmer au jalon 5bis : `id`
  auto-généré depuis le type en snake_case avec suffixe d'occurrence
  (`awgn`, `awgn_2`), surchargeable par un paramètre `name` du bloc —
  qui existe déjà partiellement (cf. défaut S6, `self.name` fantôme,
  qui trouve ici sa résolution : le champ devient réel et sert à ça).

---

## Annexe A — Défauts constatés (audit du 2026-07-28)

### A.1 Bloquants

| Emplacement | Défaut |
|---|---|
| `README.md` | `SymbolMapper(M=16)` → `TypeError` (signature réelle : `SymbolMapper(alphabet)`) ; et `tx, rx = chain(10000)` est impossible, `Sequential.forward` ne renvoie qu'une sortie — la sortie transmise s'obtient par un tap (D42) |
| `README.md` | « Only requires `numpy` and `scipy` » contredit `pyproject.toml`. À aligner sur D1 |
| `pyproject.toml` | version `0.91` vs `v0.2.0` planifié → régression PEP 440 (D9) |
| `optical/channels.py` — `PhaseNoise` | Entièrement mort : `__post_init__` lit `self.seed`, champ inexistant → `AttributeError` à la construction ; `forward` appelle `self.rvs()` au lieu de `noise_rvs()` |
| `core/compensators.py` — `TrainedBasedComplexGainCompensator` | Non fonctionnel, trois défauts cumulés : `target_data = Union[...]` utilise `=` au lieu de `:` (ce n'est pas un champ de dataclass) ; `extractor: field(default_factory=...)` place un appel `field()` dans l'emplacement du type ; `fit()` référence `x_preamble` et `N_preamble`, variables locales de `forward` → `NameError` |
| `ofdm/processors.py` — `CarrierAllocator` | Lève `TypeError` sur entrée 1D (affectation des pilotes). Résolu par D18 |

### A.2 Silencieux

| Emplacement | Défaut |
|---|---|
| `mimo/channels.py` — `BaseMIMOChannel.info()` | `H.ndims` (typo pour `ndim`) → la méthode plante systématiquement |
| `core/compensators.py` — `TrainedBasedPhaseCompensator` | `__post__init__` : underscore de trop, la méthode n'est jamais appelée |
| `ofdm/utils.py` — `get_standard_carrier_allocation` | `width = N_nulled_DC // 2` puis annulation de `[middle-width : middle+width+1]` → annule toujours un nombre **impair** de sous-porteuses (correct pour 1, 3, 11 ; faux pour 2 ou 4) |
| `ofdm/utils.py` | Paramètre `shift` : applique `fftshift` quand il vaut `False`. Résolu par D16 |
| `optical/chains.py` | Fichier vide (0 octet) |
| `core/generics.py` | `Processor.__call__` affiche `self.name` en mode debug, mais la classe de base ne définit pas de champ `name` |
| `core/generics.py` | `Sequential.__call__(x, debug=False)` : argument `debug` inutilisé |
| `core/generics.py` — `Sequential.__repr__` | `pprint.pformat(self.asdict())` : déverse les tableaux entiers (l'alphabet 16-QAM occupe huit lignes) et n'expose ni le type des blocs ni la structure. Résolu par D33a |
| `core/utils.py:47`, `core/mappers.py:59`, `ofdm/utils.py:104`, `ofdm/processors.py:496,587` | Aucune fonction de tracé n'accepte ni ne retourne un `ax` : elles prennent un numéro de figure `num=None` et possèdent leur figure. Résolu par D25 |
| `core/processors.py:910` | `plt.show()` appelé depuis `src/`. Résolu par D25 |
| `ofdm/utils.py:104` | `plot_carrier_allocation(..., color_list=["b","g","r"], label_list=[...])` : arguments par défaut **mutables**, partagés entre tous les appels |
| `ofdm/utils.py:104` vs `:119` | La signature déclare `["b","g","r"]`, la docstring annonce `["g","b","r","k"]`. L'index portant le sens, la légende documentée **inverse** null et data par rapport à la figure produite. Résolu par D27a |
| `core/visualizers.py:11` | `mpl.rcParams['agg.path.chunksize'] = 10000` au niveau module : importer comnumpy modifie l'état matplotlib global de l'utilisateur. Résolu par D27b |

### A.3 Docstrings fausses (48 doctests en échec sur 218)

| Emplacement | Défaut |
|---|---|
| `ofdm/metrics.py` — `compute_PAPR` | Documente `2.0`, calcule `1.4606` (implémentation cohérente, exemple inventé) |
| `core/metrics.py` — `compute_evm` | Documente `0.0506`, calcule `0.0365` |
| `core/metrics.py` — `compute_ccdf` | Forme de sortie documentée transposée par rapport à la sortie réelle |
| `core/generators.py` — `GaussianGenerator` | Affiche des réels issus du RNG *legacy* pour une sortie complexe |
| `mimo/utils.py` | Doctests appelant `rayleigh_iid`, `rician`, `kronecker_rayleigh` — fonctions inexistantes (renommées sans mise à jour) |
| `core/processors.py` — `Complex2Real` | Documente une `ValueError` qui n'est jamais levée |
| divers | Écarts de formatage (indentation de `print`, `np.float64(...)`) — corrigibles par `NORMALIZE_WHITESPACE` et par l'exécution effective des doctests |

### A.4 Chaîne d'intégration

| Emplacement | Défaut |
|---|---|
| `.github/workflows/tests.yml` | `python-version: "3.x"` — une seule version testée alors que le paquet annonce 3.11/3.12/3.13 |
| `.github/workflows/tests.yml` | Installe `requirements.txt`, qui tire Sphinx en CI de tests |
| `requirements.txt` | Duplique et contredit `pyproject.toml` — à supprimer au profit des extras |
| CI | Aucun doctest, aucun linter, aucune mesure de couverture |
| `examples/*` | Chemins de sauvegarde relatifs (`../../docs/...`) : les scripts n'échouent que si on ne les lance pas depuis leur propre dossier. À rendre robustes ou à documenter |
### A.5 Défauts constatés pendant la conversion §4.10 (2026-08-09)

Écrire le modèle mathématique de chaque bloc oblige à relire son code
ligne à ligne — c'est la vertu de D10 qu'on n'attendait pas. Cette passe a
levé la série ci-dessous. Ceux marqués **corrigé** l'ont été dans la même
fenêtre ; les autres attendent un arbitrage parce qu'ils changent des
résultats numériques ou un choix d'API.

| Emplacement | Défaut | État |
|---|---|---|
| `core/filters.py` — `SRRCFilter` | `method="time"` figurait dans le `Literal` sans branche dans `forward` → `UnboundLocalError` | **corrigé** (rejeté à la construction) |
| `core/filters.py` — `SRRCFilter` | `scale` appliqué par la seule voie `"fft"`, silencieusement ignoré par `"lfilter"` (le défaut) : deux méthodes censées calculer la même chose ne la calculaient pas | **corrigé** |
| `core/filters.py` — `SRRCFilter` | Paramètre `axis` mort : `-1` codé en dur dans `lfilter` | **corrigé** |
| `core/processors.py` — `BlindPhaseTracker` | Décorateur `@dataclass` manquant : la classe ne pouvait pas être construite (`TypeError`), `__post_init__` ne tournait jamais, et `forward` lisait un champ inexistant | **corrigé** |
| `core/compensators.py` — `DataAidedFIRCompensator` | Appelait `get_target_data()` sans hériter du mixin qui la définit → `AttributeError` à chaque appel | **corrigé** |
| `core/filters.py` — `BWFilter` | Le nom annonce un Butterworth ; le code applique un masque rectangulaire (passe-bas idéal, coupure infiniment raide) | documenté |
| `core/filters.py` — `BWFilter` | La docstring annonçait « 1 = Nyquist » (convention `Wn` de scipy) ; `fftfreq(N, d=1)` borne les fréquences à 1/2, donc **Nyquist vaut 0,5**. Tout appelant réglant `wn` d'après scipy filtre deux fois trop large — et `examples/optical/*` passent `1/oversampling`, ce qui suggère précisément cette lecture | **corrigé** : `wn` normalisé à Nyquist (convention scipy). `Upsampler` construisait lui-même `BWFilter(1/L)`, donc le filtre anti-image de la bibliothèque était deux fois trop large ; les sites d'appel deviennent corrects sans être touchés |
| `core/metrics.py` — `compute_ccdf` | Faux pour `ndim > 1` : `expand_dims` produit une forme qui ne se diffuse pas contre les données triées (`(3,1)` au lieu de `(1,3)`). L'ancien doctest **gravait la sortie fausse** comme si elle était voulue | **corrigé** (reshape explicite ; le chemin 1D, le seul utilisé, est inchangé) |
| `core/metrics.py` — `compute_ber` | Le paramètre `axis` n'a jamais fonctionné : `sym_2_bin` appelle `np.binary_repr`, qui exige un scalaire → `TypeError` sur une entrée 2D | documenté |
| `core/metrics.py` — `compute_ser_awgn_psk` | Branche morte `if type == "bin"` : compare le *builtin* `type` à une chaîne, la fonction n'ayant pas ce paramètre. Inoffensive aujourd'hui, dangereuse si quelqu'un la « répare » (le BER serait divisé deux fois par `k`) | à supprimer |
| `core/metrics.py` | `compute_ser_awgn_psk` (ordre hors {2,4,>4}) et `compute_metric_awgn_theo` (modulation inconnue) tombent en `UnboundLocalError` au lieu d'un `ValueError` explicite (D38) | à corriger |
| `core/processors.py` — `Downsampler` | `use_filter=True` → `AttributeError` : le champ `filter` n'est pas déclaré, contrairement à `Upsampler` (et `slots=True` rend l'absence immédiate) | documenté |
| `core/processors.py` — `Amplifier` | Le paramètre `axis` n'implémente aucun modèle défendable : `[1]*ndim` avec le gain déposé à une position, diffusé contre `X` — il n'amplifie que les entrées d'indice `axis` du **dernier** axe. L'ancien exemple gravait ce comportement | documenté, `WeightAmplifier` recommandé |
| `core/processors.py` — `Clipper` | Docstring et code en désaccord : le modèle décrivait un écrêtage polaire, le code fait `np.clip(x, -τ, τ)`, qui sur complexe compare lexicographiquement | docstring alignée sur le code |
| `optical/compensators.py` — `ChromaticDispersionLSFIRCompensator` | `q_col` alloué à zéro et jamais rempli : `toeplitz` rend une matrice triangulaire supérieure au lieu de la matrice de Gram hermitienne. Masqué à la bande par défaut où `q_row[m] = sinc(m) = 0` | **corrigé** |
| `optical/compensators.py` — `ChromaticDispersionLSFIRCompensator` | `d_vect` ignore `w_vect` : la forme fermée en `erf` est celle de la bande pleine `[-π, π]`. Conjugué au point précédent, `w_vect` est inutilisable ailleurs qu'à sa valeur par défaut | **corrigé** (forme fermée générale par complétion du carré ; la bande n'entre que par les deux bornes d'erf) |
| `optical/compensators.py` — `ChromaticDispersionLSFIRCompensator` | **Troisième défaut, invisible tant que les deux premiers subsistaient.** Avec un `Q` correct, une bande réduite est réellement déficiente en rang (spectre prolate : environ `N(Ω₂−Ω₁)/2π` valeurs propres utiles, le reste à 1e-16). `LA.inv(Q + εI) @ d` rendait alors des coefficients de 3,2e9 ; `LA.solve` sur le même système est exact. Mathématiquement la même opération, donc la bande par défaut ne bouge pas | **corrigé** |
| `core/processors.py` — `SampleRemover`, `DataAdder` | 1D seulement (`len(x)`, `concatenate` sur l'axe 0) sans garde : une entrée 2D échoue obscurément au lieu de lever `ShapeError` (D38) | à corriger |
| `ofdm/utils.py` — `plot_carrier_allocation` | Couleurs `"b"/"g"/"r"` en dur, alors que D27 impose `CARRIER_STYLE` (Okabe-Ito, la couleur jamais seule porteuse d'information) | à corriger |
| `ofdm/utils.py` — `plot_carrier_allocation` | L'exemple était un bloc markdown dans une docstring numpydoc : jamais rendu, jamais exécuté | **corrigé** (doctest réel) |
| `core/compensators.py` — `DataAidedFineSynchronizer` | `forward` lisait `self.up_factor` en quatre endroits, champ jamais déclaré → `AttributeError` à chaque appel. Classe entièrement non fonctionnelle | **corrigé** (champ ajouté, défaut 2 à confirmer) |
| `core/compensators.py` — les deux synchroniseurs | La correction d'amplitude semble **inversée** : `scale = c[m̂₀] ≈ a·E[|d|²]` est appliquée par *multiplication*, donc une atténuation `a` ressort en `a²·E[|d|²]` — l'altération est amplifiée au lieu d'être compensée. Mesuré : sur un gain de `0,5·exp(0,3j)`, la sortie valait 0,25× la référence | **corrigé** (restitution exacte, erreur 0,0) |
| `core/compensators.py` — `BlindIQCompensator` | L'implémentation GSOP de Fatadin *et al.* est présente sous forme de littéral de chaîne entre deux méthodes : du code mort. Le `fit` actif est un blanchiment par décomposition propre de la covariance réelle 2×2, pas une orthogonalisation de Gram-Schmidt. De plus `np.linalg.eig` ne trie pas ses vecteurs propres : I et Q peuvent ressortir permutés ou changés de signe (les statistiques d'ordre 2 sont invariantes, la constellation non) | documenté |
| `core/compensators.py` — `Normalizer` | Hérite d'`Amplifier`, donc la vraie signature est `Normalizer(gain, method, ...)` : `Normalizer('max')` affecte silencieusement `gain='max'` et laisse `method='amp'` | documenté |
| `core/compensators.py` | Violations de D23 (underscore final sur les grandeurs estimées) : `Normalizer.gain`, `BlindIQCompensator.alpha/beta`, `BlindCFOCompensator.w0`, `DataAidedFIRCompensator.h`, `delay`/`scale`/`cross_corr` des synchroniseurs. Seuls `theta_` et `gain_` s'y conforment | à corriger |
| `core/compensators.py` — `BlindCFOCompensator` | `should_fit=False` sans `fit` préalable donne `w0 = None` → `TypeError` dans `np.exp` au lieu du `NotFittedError` qu'exige D23 | à corriger |
| `core/compensators.py` — `DCCorrector` | `axis=0` par défaut, ce qui contredit D2 (les échantillons sont sur le dernier axe) | documenté |
| `mimo/utils.py` — `apply_correlation` | Cassée et fausse : `if Rx:` teste la véracité d'un tableau → `ValueError` pour toute matrice réelle ; `Rx` et `Ry` sont tous deux appliqués **à droite**, donc `Ry` ne peut pas représenter la corrélation en réception ; et `H @ L` donne `L^H L ≠ R` (il faudrait `L.T`, ce que `kronecker_rayleigh_channel` fait correctement). Inutilisée dans tout le dépôt et non exportée | **candidate à la suppression** |
| `mimo/utils.py` — `rician_channel` | `np.empty_like(H_los)` fait hériter le dtype de la composante LoS : avec un `H_los` réel, la partie imaginaire de la composante diffuse est **silencieusement supprimée** (simple `ComplexWarning`) | à corriger |
| `core/utils.py` + `core/metrics.py` | `sym_2_bin` existe en double, corps identiques, défauts différents (`width=4` d'un côté, obligatoire de l'autre) ; `compute_ber` utilise la copie de `metrics.py` | à dédupliquer |
| `core/data/*.csv` | En-tête `s,bin,real,imag` (4 noms) pour des lignes à 3 colonnes. `get_alphabet` lit les colonnes 1 et 2 et ignore la 0, donc la correspondance repose sur le tri des lignes par indice de symbole | en-tête trompeur, à corriger |
| `core/utils.py` — `mmse_estimator` | Le régularisateur `σ²I` n'est correct que pour un alphabet d'énergie unitaire ; la forme générale est `(σ²/E_s)I`. Cohérent avec `get_alphabet(norm=True)`, silencieusement faux avec `norm=False` | documenté |
