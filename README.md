# tram_optimizer

Aplikacja wspierająca decyzje dyspozytorskie w zakresie optymalizacji rozkładu jazdy tramwajów. System realizuje dwufazowy proces planowania:

1. **Faza 1** — Symulowane Wyżarzanie wyznacza optymalną liczbę kursów w każdej godzinie doby, minimalizując czas oczekiwania pasażerów i koszt floty.
2. **Faza 2** — algorytm dystrybuanty CDF przekształca częstotliwości godzinowe w minutowy rozkład odjazdów, wygładzając przejścia między okresami o różnym popycie.

Główny interfejs to notatnik Jupyter (`app.ipynb`). Logika obliczeniowa znajduje się w modułach Pythona w katalogu głównym projektu.

## Wymagania

- Python 3.10+
- Do generowania plakatów PDF: biblioteki systemowe dla [WeasyPrint](https://doc.courtbouillon.org/weasyprint/stable/first_steps.html)

## Instalacja

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Uruchomienie

1. Uruchom Jupyter i otwórz notatnik aplikacji:

```bash
jupyter notebook app.ipynb
```

2. Uruchamiaj komórki kolejno (`Shift + Enter`).

3. W pierwszej komórce możesz wygenerować dane testowe (`generate_dummy_data`) lub umieścić własne pliki CSV w `data/input/`:
   - `travel_times.csv` — czasy przejazdu między przystankami (`Origin`, `Destination`, `TravelTime_min`)
   - `demand.csv` — popyt pasażerski (`Origin_s`, `Destination_d`, `TimeWindow_t`, `Demand_P`)
   - `stop_names.csv` — nazwy przystanków (`stop_id`, `stop_name`)

4. Skonfiguruj parametry algorytmu i modelu w kolejnych komórkach, uruchom optymalizację (Faza 1) i generowanie rozkładu (Faza 2).

5. Wyniki trafiają do:
   - `data/output/final_timetable.csv` — harmonogram w formacie tabelarycznym
   - `data/output/pdf/` — plakaty przystankowe w PDF

## Inne notatniki i skrypty

| Plik | Opis |
|------|------|
| `experiments.ipynb` | strojenie hiperparametrów Symulowanego Wyżarzania |
| `tester.py` | analiza wrażliwości parametrów modelu (`C`, `R`, `α`, `β`) |
| `Example_usage.ipynb` | przykładowe użycie modułów |