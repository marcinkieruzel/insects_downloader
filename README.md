# Zbiór zdjęć europejskich owadów

Powtarzalny pipeline, który pobiera zdjęcia europejskich owadów z publicznego
[iNaturalist Open Dataset](https://github.com/inaturalist/inaturalist-open-data)
na AWS S3 i buduje z nich zbalansowaną klasowo strukturę katalogów gotową do
trenowania modelu. Repozytorium zawiera również wytrenowany klasyfikator
(EfficientNet-B4) oraz prostą aplikację Flask udostępniającą model przez API
i przeglądarkowy formularz.

Cel: ~80 GB zdjęć w rozdzielczości 240 px obejmujących tysiące europejskich
gatunków owadów — wystarczające do wytrenowania klasyfikatora obrazu pracującego
na wejściu 224 px.

## Instalacja

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` zawiera wszystkie zależności potrzebne zarówno do pipeline'u
danych, jak i do uruchomienia aplikacji Flask (Flask, OpenVINO, PyTorch,
torchvision).

## Kolejność uruchamiania pipeline'u (model jest już wytrenowany, nie trzeba powtarzać treningu aby po prostu zobaczyć efekt, przejdź do kolejnego kroku)

```bash
python src/1_download_metadata.py   # ~12 GB plików .csv.gz + shapefile Natural Earth
python src/2_filter.py              # tworzy filtered/photos_to_download.parquet, usuwa surowe metadane
python src/3_download_photos.py     # pobiera do ~80 GB zdjęć, wielowątkowo, z wznawianiem
python src/4_build_manifest.py      # weryfikuje zdjęcia, zapisuje manifest.csv + attribution.csv + class_counts.csv
python src/5_train_test_split.py    # stratyfikowany podział 80/10/10 do plików CSV w splits/
```

## Aplikacja Flask (klasyfikator)

Po wytrenowaniu modelu (lub korzystając z dostarczonych `checkpoints/best.pt`)
można uruchomić prostą aplikację webową, która udostępnia inferencję na CPU
przez OpenVINO.

### Uruchomienie

```bash
.venv/bin/python www/app.py
```

Przy pierwszym uruchomieniu skrypt jednorazowo konwertuje `checkpoints/best.pt`
do formatu OpenVINO IR (`checkpoints/best.xml` + `best.bin`) i cache'uje wynik
na dysku — kolejne starty są natychmiastowe. Domyślnie inferencja działa na
CPU; aby użyć zintegrowanej karty Intel, zmień w pliku
[www/app.py](www/app.py) wartość `DEVICE = "CPU"` na `"GPU"`.

Po starcie aplikacja nasłuchuje pod adresem **http://127.0.0.1:5000/**.

### Dostępne endpointy

- **`GET /`** — strona HTML z formularzem do wgrania zdjęcia. Po wybraniu
  pliku pokazuje miniaturę i tabelę z top-5 predykcjami (nazwa gatunkowa +
  prawdopodobieństwo).
- **`POST /predict`** — czyste API JSON. Przyjmuje `multipart/form-data` z
  polem `image`, zwraca top-5 predykcji:

  ```bash
  curl -F "image=@/sciezka/do/owada.jpg" http://127.0.0.1:5000/predict
  ```

  Przykładowa odpowiedź:

  ```json
  {
    "predictions": [
      {"taxon_id": 338651, "scientific_name": "Mantispa styriaca", "score": 0.9957},
      {"taxon_id": 496767, "scientific_name": "Chorthippus apricarius", "score": 0.0001},
      ...
    ]
  }
  ```

Model wykorzystuje też lekki test-time augmentation (uśrednienie predykcji dla
oryginalnego i odbitego horyzontalnie obrazu), więc każde żądanie wykonuje dwa
przejścia przez sieć.

### Konfiguracja przez zmienne środowiskowe

| Zmienna         | Domyślnie   | Opis |
|-----------------|-------------|------|
| `HOST`          | `127.0.0.1` | Adres bind dev servera (Flask). |
| `PORT`          | `5000`      | Port dev servera. |
| `DEVICE`        | `CPU`       | Urządzenie OpenVINO (`CPU` lub `GPU`). |
| `MAX_UPLOAD_MB` | `20`        | Maks. rozmiar uploadu; po przekroczeniu zwracany jest `413` z JSON-em. |
| `LOG_LEVEL`     | `INFO`      | Poziom loggera (`DEBUG` / `INFO` / `WARNING` / ...). |

Dodatkowo dostępny jest endpoint **`GET /healthz`** zwracający
`{"status":"ok","num_classes":N,"device":"CPU"}` — używany przez load balancery
oraz systemd do health checków.

### Deploy produkcyjny (gunicorn + systemd)

Plik konfiguracyjny: [deploy/insects-classifier.service](deploy/insects-classifier.service).
Zakłada on instalację w `/opt/insects` (z venv w `/opt/insects/.venv`)
i użytkownika systemowego `insects`.

```bash
# 1. Skopiuj repo do /opt/insects, utwórz venv i zainstaluj zależności.
sudo mkdir -p /opt/insects && sudo chown $USER /opt/insects
rsync -a --exclude=.venv --exclude=inat_metadata --exclude=data ./ /opt/insects/
cd /opt/insects
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 2. Utwórz użytkownika i ustaw właściciela.
sudo useradd --system --home /opt/insects --shell /usr/sbin/nologin insects
sudo chown -R insects:insects /opt/insects

# 3. Zainstaluj i uruchom usługę.
sudo cp deploy/insects-classifier.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now insects-classifier
sudo systemctl status insects-classifier
```

Domyślnie usługa nasłuchuje na `0.0.0.0:8000` (zmień `Environment=PORT=8000`
i `HOST=` w pliku unit). Gunicorn działa z `--workers 1 --threads 4 --preload`,
czyli **jedna kopia modelu w RAM** (ok. 300 MB) i obsługa wielu równoczesnych
żądań przez wątki — optymalne dla CPU inference (OpenVINO uwalnia GIL).
`--timeout 120` pokrywa jednorazową konwersję PT→IR przy pierwszym starcie.

Logi gunicorna idą na stdout/stderr — przeglądasz je przez:

```bash
journalctl -u insects-classifier -f
```

Przed wystawieniem na świat warto postawić przed gunicornem reverse-proxy
(nginx/Caddy) zapewniający TLS i ewentualny rate limiting.

## Konfiguracja pipeline'u

Wszystkie parametry znajdują się w pliku [config.yaml](config.yaml).
Najważniejsze:

- `size: small` — 240 px. Przełącz na `medium` (500 px) tylko jeśli zwiększysz `disk_budget_gb`.
- `min_photos_per_species: 20` — odrzuca gatunki z mniejszą liczbą zdjęć.
- `max_photos_per_species: 500` — ogranicza zbyt liczne klasy (losowanie z ziarnem).
- `disk_budget_gb: 80` — twardy limit w skrypcie 3, który zatrzymuje pobieranie po osiągnięciu rozmiaru.
- `licenses_allowed` — domyślnie wszystkie warianty Creative Commons dostępne w iNaturalist.

## Budżet dyskowy

Pipeline mieści się w ok. 92 GB w szczycie:

- Krok 1 zapisuje ~12 GB metadanych do `inat_metadata/`.
- Krok 2 kasuje `inat_metadata/` po filtrowaniu (gdy `delete_metadata_after_filter: true`).
- Krok 3 zapisuje do `disk_budget_gb` GB zdjęć w `data/images/`.
- Krok 3 sprawdza rozmiar `data/images/` co 5 000 pobrań i kończy pracę po przekroczeniu budżetu.

## Wznawianie

Wszystkie skrypty są idempotentne:

- Skrypt 1 pomija pliki metadanych, które już istnieją i przechodzą test integralności gzip.
- Skrypt 3 pomija zdjęcia obecne już na dysku z rozmiarem > 0.
- Skrypty 2, 4, 5 po prostu nadpisują swoje wyniki.

Możesz przerwać dowolny krok przez Ctrl+C i uruchomić go ponownie — żadna praca nie jest powtarzana.

## Modyfikacja zbioru danych

- **Powiększenie**: zwiększ `max_photos_per_species` i ponownie uruchom skrypty 2 + 3. Pobierane są tylko nowe pliki.
- **Zmniejszenie**: zmniejsz limit i ręcznie usuń nadmiarowe pliki — skrypt nie czyści automatycznie.
- **Inny obszar geograficzny**: zmień `bbox` i `use_precise_europe_polygon`, ponownie uruchom skrypty 2 + 3.

## Struktura wyjścia

```
data/
└── images/
    └── <taxon_id>_<slug_nazwy_gatunkowej>/
        └── <photo_id>.<ext>

manifest.csv         # jeden wiersz na zdjęcie z metadanymi
attribution.csv      # podpis CC dla każdego zdjęcia
class_counts.csv     # liczba zdjęć na gatunek
splits/{train,val,test}.csv

checkpoints/
├── best.pt          # wytrenowane wagi PyTorch
├── best.xml         # OpenVINO IR (generowane przy pierwszym starcie Flask)
└── best.bin
```

## Etyka

Większość zdjęć z iNaturalist jest na licencji CC-BY-NC.
**Nie wolno redystrybuować ich w celach komercyjnych.**
Zawsze dołączaj `attribution.csv` razem ze zbiorem danych.

## Źródła danych

- Metadane iNaturalist Open Dataset: https://inaturalist-open-data.s3.amazonaws.com/
- Adresy zdjęć: `https://inaturalist-open-data.s3.amazonaws.com/photos/{photo_id}/{size}.{extension}`
- Granice państw Natural Earth (do wyznaczenia poligonu Europy): https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip
