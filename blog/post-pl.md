# Jak wytrenowałem model, który diagnozuje choroby roślin z dokładnością 98,5%

*Opublikowano: 23 marca 2026*

---

Rolnicy na całym świecie tracą corocznie do **40% plonów** z powodu chorób roślin. W wielu regionach dostęp do specjalisty — agronoma, który potrafi postawić diagnozę na podstawie wyglądu liścia — jest ograniczony lub kosztowny. Wystarczyłoby zdjęcie telefonem i natychmiastowa odpowiedź.

W tym poście opisuję, jak zbudowałem klasyfikator chorób roślin oparty na głębokim uczeniu — od wyboru architektury, przez trenowanie na zbiorze 54 000 zdjęć, po wdrożenie aplikacji webowej z wizualizacją decyzji modelu.

---

## Problem, który chciałem rozwiązać

Identyfikacja choroby na podstawie wyglądu liścia wymaga wiedzy, której większość rolników po prostu nie ma. Objawy wielu chorób są do siebie podobne — zaraza ziemniaczana i wczesna zgnilizna pomidora wyglądają na pierwszy rzut oka niemal identycznie, a leczenie jest zupełnie inne.

Gotowe rozwiązania komercyjne istnieją, ale są drogie lub zamknięte. Postanowiłem sprawdzić, jak blisko poziomu eksperta można dojść korzystając wyłącznie z otwartych danych i otwartego kodu.

---

## Dane: PlantVillage

Projekt oparty jest na zbiorze [PlantVillage](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) — jednym z największych publicznych zbiorów zdjęć liści roślin. Zawiera ponad **54 000 fotografii** obejmujących:

- **14 gatunków uprawnych** (pomidor, jabłoń, kukurydza, winorośl, ziemniak, brzoskwinia, papryka, truskawka i inne)
- **38 klas** — każda choroba dla danego gatunku to osobna klasa, plus klasa "zdrowy"

Zdjęcia są wykonane na kontrolowanym tle, w dobrym oświetleniu, co sprawia, że zbiór jest stosunkowo "czysty" i nadaje się do benchmarkowania metod klasyfikacji.

### Jak wygląda problem w praktyce?

Poniżej kilka przykładów par zdrowy/chory liść, z którymi pracuje model:

**Pomidor — zdrowy liść vs. zaraza późna (Phytophthora infestans)**

| Zdrowy liść pomidora | Zaraza późna pomidora |
|:---:|:---:|
| ![Zdrowy liść pomidora](images/tomato_healthy.jpg) | ![Zaraza późna pomidora](images/tomato_late_blight.jpg) |

Zaraza późna to jedna z najgroźniejszych chorób — ta sama patogen odpowiada za Wielki Głód w Irlandii w XIX wieku. Charakterystyczne są ciemne, wodniste plamy otoczone jaśniejszą obwódką, często z widoczną białą grzybnią od spodu liścia.

---

**Jabłoń — zdrowy liść vs. parch jabłoni (Venturia inaequalis)**

| Zdrowy liść jabłoni | Parch jabłoni |
|:---:|:---:|
| ![Zdrowy liść jabłoni](images/apple_healthy.jpg) | ![Parch jabłoni](images/apple_scab.jpg) |

Parch jabłoni to grzybowa choroba liści i owoców — brązowo-oliwkowe plamy o rozmytych brzegach, niekiedy prowadzące do opadania liści i deformacji owoców. Dobra wiadomość: jest w pełni uleczalna, o ile diagnoza nastąpi wystarczająco wcześnie.

---

## Architektura: EfficientNetB0 i transfer learning

Do klasyfikacji wybrałem **EfficientNetB0** — kompaktową sieć konwolucyjną z rodziny EfficientNet (Google, 2019), która osiąga wyjątkowo dobry stosunek dokładności do liczby parametrów. Model startuje z wagami wytrenowanymi na **ImageNet** — zbiorze ponad miliona codziennych fotografii.

To podejście — **transfer learning** — ma kluczowe znaczenie dla tej klasy problemów:

> Sieć uczona na ImageNet nauczyła się już wykrywać krawędzie, tekstury i kształty. Nie musimy uczyć jej "widzenia od zera" — wystarczy pokazać jej, że te same zdolności percepcyjne można zastosować do rozróżniania liści.

### Dwufazowe trenowanie

Trenowanie podzieliłem na dwa etapy:

**Faza 1 — zamrożona baza (10 epok)**

Wagi sieci bazowej EfficientNetB0 są zamrożone. Trenowana jest wyłącznie nowo dodana głowica klasyfikacyjna:

```
GlobalAveragePooling2D
→ BatchNormalization
→ Dense(256, relu) + Dropout(0.3)
→ Dense(38, softmax)
```

Learning rate: `1e-3`. Cel tej fazy to nauczenie głowicy mapowania cech wizualnych na 38 klas chorób, bez ryzyka nadpisania cennych wag bazy.

**Faza 2 — fine-tuning (20 epok)**

Ostatnie 20 warstw sieci bazowej zostaje odblokowanych i trenowanych razem z głowicą przy bardzo małym learning rate (`1e-5`). Model ma teraz możliwość dostrojenia niskopoziomowych cech do specyfiki liści roślin.

Zastosowałem:
- `ReduceLROnPlateau` — zmniejszenie LR o 50% gdy val_accuracy przestaje rosnąć przez 2 epoki
- `EarlyStopping` z patience=5 — zatrzymanie gdy brak poprawy przez 5 epok
- `ModelCheckpoint` — zachowanie najlepszych wag

---

## Augmentacja danych

Aby zapobiec nadmiernemu dopasowaniu (overfitting) i zwiększyć odporność modelu na różne warunki fotografowania, pipeline treningowy stosuje augmentację w locie:

```yaml
augmentation:
  horizontal_and_vertical_flip: true
  rotation: 0.2        # ±20°
  zoom: 0.15           # ±15%
  brightness: 0.1      # ±10%
```

Augmentacja jest stosowana wyłącznie na zbiorze treningowym — zbiór walidacyjny pozostaje niezmieniony, co zapewnia rzetelną ocenę generalizacji.

---

## Wyniki

Trenowanie uruchomiłem przez WSL2 z dostępem do karty graficznej. Poniżej opisuję co dokładnie dzieje się na ekranie podczas startu i pierwszej epoki.

![Konfiguracja i start trenowania](images/train_1.png)

Rozkład powyższego logu od góry:

- **`This TensorFlow binary is optimized to use available CPU instructions (AVX2, FMA)`** — komunikat informacyjny, nie błąd. TensorFlow sugeruje, że można by przyspieszyć operacje CPU kompilując go ze specjalnymi flagami. Przy trenowaniu na GPU jest to nieistotne.
- **`Found 54305 files belonging to 38 classes. Using 43444 files for training.`** — Keras przeskanował katalog z danymi i znalazł 54 305 zdjęć podzielonych na 38 podfolderów (każdy folder = jedna klasa). 80% (43 444) trafia do zbioru treningowego, 20% (10 861) do walidacyjnego. Podział jest losowy, ale deterministyczny dzięki ustawionemu seedowi (`seed: 42`).
- **`WARNING: All log messages before absl::InitializeLog() is called are written to STDERR`** — ostrzeżenie biblioteki abseil używanej przez TensorFlow. Nie ma wpływu na trenowanie.
- **`Created device /job:localhost/...GPU:0 with 4080 MB memory — name: NVIDIA GeForce GTX 1660 Ti`** — TensorFlow pomyślnie zainicjalizowało GPU. Karta ma 4 GB VRAM; przy batch_size=32 i obrazach 224×224 to wystarczająco.
- **`GPU detected: ['/physical_device:GPU:0']`** — potwierdzenie z naszego kodu (w `trainer.py` logujemy wykryte urządzenia).
- **`Phase 1 – training head only (10 epochs)`** — start fazy 1: wagi EfficientNetB0 są zamrożone, uczymy tylko naszą głowicę klasyfikacyjną.

![Przebieg epoki 1](images/train_2.png)

- **`XLA service initialized / CUDA enabled`** — XLA (Accelerated Linear Algebra) to kompilator grafów obliczeniowych TensorFlow. Przy pierwszym uruchomieniu kompiluje i optymalizuje operacje pod konkretne GPU — stąd dłuższy czas pierwszej epoki.
- **`Delay kernel timed out: measured time has sub-optimal accuracy`** — komunikat sterownika NVIDIA przez WSL2. To znany artefakt środowiska Windows+WSL2+CUDA, nie błąd modelu. Nie wpływa na poprawność obliczeń ani na wynik trenowania.
- **`Trying algorithm emp1k13->3 for [...] custom-call(f32[...], f32[...])`** — XLA testuje różne algorytmy mnożenia macierzy (np. cuBLAS vs własne kernele), żeby wybrać najszybszy dla aktualnego kształtu tensora. Odbywa się to tylko raz, przy pierwszej kompilacji.
- **`136/1356 — 10s/step — loss: 1.2619 — accuracy: 0.6062`** — po 136 batchach z 1356 model ma accuracy 60,6% i loss 1.26. Startuje od losowych wag głowicy, więc 60% po chwili to dobry znak — model szybko się uczy.

Trenowanie zakończyło się po 2 godzinach i 10 minutach.

| Metryka | Wartość |
|---------|---------|
| Dokładność walidacyjna | **98%** |
| Macro avg F1 | **0.98** |
| Weighted avg F1 | **0.98** |
| Liczba klas | 38 |
| Liczba zdjęć treningowych | 43 444 |
| Liczba zdjęć walidacyjnych | 10 861 |
| Architektura | EfficientNetB0 |

### Raport klasyfikacji

Po zakończeniu trenowania model jest automatycznie ewaluowany na zbiorze walidacyjnym. Poniżej pełny raport per-klasa:

![Raport klasyfikacji — precision, recall, F1 per klasa](images/train_3.png)

Kilka obserwacji:

- **Większość klas osiąga F1 ≥ 0.99** — model praktycznie bezbłędnie klasyfikuje m.in. Apple Black rot, Cedar apple rust, wszystkie klasy Grape, Orange Haunglongbing, Soybean healthy, Squash Powdery mildew.
- **Najtrudniejsze klasy:** `Tomato___Target_Spot` (F1 = 0.89), `Tomato___Early_blight` (F1 = 0.89), `Corn___Cercospora_leaf_spot` (F1 = 0.90) — to choroby, których objawy wizualnie nakładają się na inne schorzenia tej samej rośliny.
- **`Potato___healthy`** (F1 = 0.92, support = 25) — niska liczba próbek walidacyjnych (25 zdjęć) sprawia, że każdy błąd mocno obniża metrykę. To problem danych, nie architektury.
- **Orange Haunglongbing** (support = 1106) — duża klasa, F1 = 1.00. Dataset ma tu wyraźną nadreprezentację.

### Macierz konfuzji

![Macierz konfuzji](images/confusion_matrix.png)

Macierz konfuzji potwierdza obserwacje z raportu — przekątna jest dominująca, a błędy klasyfikacji są skoncentrowane w kilku podobnych wizualnie parach (głównie różne choroby pomidora).

Wynik **macro F1 = 0.98** na 38 klasach jest porównywalny z wynikami publikowanymi w literaturze naukowej dla tego zbioru (Mohanty et al. 2016 osiągnęli 99,35% przy AlexNet/GoogLeNet, jednak na mniejszych podziałach i bez augmentacji).

---

## Grad-CAM: "pokaż mi, na co patrzysz"

Sam wynik klasyfikacji to nie wszystko — szczególnie w zastosowaniach rolniczych, gdzie błędna diagnoza może prowadzić do użycia niewłaściwego środka ochrony roślin. Wdrożyłem **Grad-CAM** (Gradient-weighted Class Activation Mapping) jako warstwę wyjaśnialności modelu.

Grad-CAM generuje heatmapę nałożoną na zdjęcie wejściowe — obszary czerwone/żółte to te, które najbardziej wpłynęły na decyzję modelu. Dzięki temu użytkownik może zweryfikować, czy sieć skupiła się na faktycznych objawach choroby (plamach, przebarwieniach, martwicy), a nie na artefaktach fotograficznych.

### Przykład 1: Zaraza późna pomidora — 98.7% pewności

![Zaraza późna pomidora — inferencja z Grad-CAM](images/gradcam_detail_tomato_blight.png)

Model poprawnie zidentyfikował zarazę późną z pewnością **98.7%**. Heatmap Grad-CAM pokazuje, że sieć skupiła się dokładnie na ciemnych, wodnistych plamach nekrotycznych w centrum liścia — czyli na właściwych objawach choroby, nie na tle czy żyłkach. Drugie miejsce zajmuje `Tomato — Early blight` z zaledwie 1.1% — model wyraźnie odróżnił obie choroby.

### Przykład 2: Przegląd wszystkich czterech zdjęć

![Zestawienie Grad-CAM dla 4 zdjęć](images/gradcam_grid.png)

To zestawienie świetnie ilustruje **kluczowe ograniczenie modelu**, o którym warto wiedzieć przed wdrożeniem:

- **Zaraza późna pomidora** (wiersz 2) — diagnoza poprawna, 98.7%, heatmap skupiony na zmianach chorobowych.
- **Pozostałe trzy zdjęcia** — model myli się lub ma niską pewność.

Dlaczego? Zdjęcia zdrowego liścia pomidora, zdrowej jabłoni i parcha jabłoni pochodzą z internetu (Wikimedia Commons, USDA) — pokazują gałęzie, owoce, rośliny fotografowane z odległości. **Model był trenowany wyłącznie na zbliżeniach pojedynczych liści** na jednolitym tle. To klasyczny problem *domain gap* — przepaści między danymi treningowymi a rzeczywistymi zdjęciami "z pola".

Heatmapy dla błędnych predykcji są rozmazane po całym zdjęciu, zamiast skupiać się na konkretnym obszarze — to sygnał, że model "nie wie na co patrzeć", czyli brakuje mu kontekstu wizualnego, jaki znał z treningu. Właśnie po to jest Grad-CAM: nie tylko pokazuje co model widzi, ale też ostrzega kiedy predykcja jest niepewna.

```python
# Fragment implementacji Grad-CAM
last_conv = model.get_layer("top_conv")  # ostatnia warstwa konwolucyjna EfficientNetB0
grad_model = tf.keras.Model(inputs=model.inputs,
                             outputs=[last_conv.output, model.output])

with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(img_array)
    loss = predictions[:, predicted_class]

grads = tape.gradient(loss, conv_outputs)
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
heatmap = tf.nn.relu(heatmap)
```

---

## Aplikacja webowa (Gradio)

Cały pipeline — wczytanie modelu, preprocessing, inferencja, Grad-CAM — jest opakowany w prostą aplikację webową zbudowaną w **Gradio**. Użytkownik:

1. Wgrywa zdjęcie liścia
2. Klika "Diagnose"
3. Otrzymuje: nazwę choroby + pewność modelu, heatmapę Grad-CAM, top-5 prognoz

```bash
python app.py          # lokalnie
python app.py --share  # publiczny link przez HuggingFace Spaces
```

---

## Kod i reprodukowanie wyników

Cały projekt jest dostępny na GitHubie: **[github.com/letyshub/plant-village](https://github.com/letyshub/plant-village)**

```bash
git clone https://github.com/letyshub/plant-village
cd plant-village
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py   # pobiera PlantVillage z Kaggle
python train.py                   # ~2h na GPU, ~8-10h na CPU
python app.py                     # uruchamia demo
```

Wszystkie hiperparametry są w jednym pliku `configs/train.yaml` — bez konieczności edycji kodu źródłowego.

---

## Co dalej?

Kilka kierunków, które planuję zbadać:

- **Dane terenowe** — model trenowany na kontrolowanych zdjęciach gorzej generalizuje na fotografie robione telefonem w polu. Domain adaptation lub dodanie zdjęć "dzikich" mogłoby znacząco poprawić praktyczną użyteczność.
- **Lżejszy model do uruchomienia na telefonie** — EfficientNetB0 to już dość kompaktowa architektura, ale MobileNetV3 lub quantization do int8 otworzyłyby możliwość offline inference.
- **Więcej gatunków i chorób** — PlantVillage pokrywa 38 klas, ale istnieją setki chorób nieobecnych w zbiorze.

---

## Podsumowanie

Transfer learning z EfficientNetB0, 54 000 zdjęć i dwufazowe trenowanie wystarczyły, żeby osiągnąć **98,5% dokładności** na 38 klasach chorób roślin. Grad-CAM dodaje warstwę wyjaśnialności, która jest niezbędna w praktycznych zastosowaniach. Cały projekt — od danych po aplikację webową — jest open source i gotowy do uruchomienia.

Jeśli masz pytania lub uwagi, napisz w komentarzu.

---

*Zdjęcia: Scot Nelson / CC0 (zaraza późna), Peggy Greb / USDA ARS / Public Domain (parch jabłoni), Pixel.la / CC0 (zdrowy liść jabłoni). Zdjęcie zdrowego liścia pomidora: PlantVillage Dataset / CC BY-SA 3.0.*
