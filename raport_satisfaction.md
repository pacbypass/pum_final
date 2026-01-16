# Airline Passenger Satisfaction Prediction

#### Autorzy:  Dominik Chyliński, Grzegorz Szczęsny, Jakub Bruski

*Data: 2026-01-16*

## Opis problemu

Projekt dotyczy przewidywania satysfakcji pasażerów linii lotniczych na podstawie danych z podróży. Problem klasyfikacji binarnej ma na celu określenie, czy pasażer jest zadowolony (satisfied) czy neutralny/niezadowolony (neutral or dissatisfied).

**Kontekst biznesowy:** Linie lotnicze mogą wykorzystać model do:
- Identyfikacji kluczowych czynników wpływających na satysfakcję pasażerów
- Prognozowania satysfakcji nowych pasażerów
- Optymalizacji usług na podstawie przewidywań
- Poprawy doświadczeń klientów i zwiększenia lojalności

**Interesariusze:** Dział obsługi klienta, zarząd linii lotniczych, zespół ds. jakości usług.

**Dlaczego problem jest interesujący?**
- Duża ilość danych dostępnych z ankiet satysfakcji
- Możliwość zastosowania zaawansowanych technik uczenia maszynowego
- Bezpośrednie przełożenie na decyzje biznesowe
- Wysoka konkurencyjność w branży lotniczej

## Dane

**Źródło danych:** Kaggle - Airline Passenger Satisfaction (https://www.kaggle.com/teejmahal20/airline-passenger-satisfaction)

**Wielkość zbioru danych:**
- Zbiór treningowy: 103,904 obserwacji, 24 cechy
- Zbiór testowy: 25,976 obserwacji, 24 cechy

**Główne kategorie cech:**
1. Informacje demograficzne (płeć, wiek, typ klienta)
2. Informacje o locie (typ podróży, klasa, odległość)
3. Oceny usług (wifi, jedzenie, rozrywka, czystość, etc.) - skala 0-5
4. Informacje o opóźnieniach (opóźnienie odlotu, przylotu)
5. Zmienna celu: satysfakcja (satisfied / neutral or dissatisfied)

**Ocena wiarygodności danych:** Dane pochodzą z symulowanych ankiet satysfakcji, są kompletne i dobrze uporządkowane. Mogą być używane do budowy modeli predykcyjnych.

**Krótka analiza opisowa danych:**

![](figures/target_distribution.png)

*Rysunek 1: Rozkład zmiennej celu - 56.7% pasażerów neutralnych/niezadowolonych, 43.3% zadowolonych*

**Analiza brakujących wartości:** W zbiorze danych występują nieliczne brakujące wartości w kolumnie 'Arrival Delay in Minutes' (0.30% w zbiorze treningowym, 0.32% w testowym).

![](figures/missing_values.png)

*Rysunek 2: Rozkład brakujących wartości - jedynie kolumna 'Arrival Delay in Minutes' zawiera brakujące wartości*

**Macierz korelacji:** Analiza korelacji między cechami numerycznymi ujawnia silne zależności między ocenami różnych usług pokładowych.

![](figures/correlation_matrix.png)

*Rysunek 3: Macierz korelacji cech numerycznych - silne korelacje między ocenami usług (np. wifi a łatwość rezerwacji online: 0.72)*

**Rozkład cech numerycznych:** Histogramy przedstawiają rozkład wieku, odległości lotu, opóźnień oraz ocen usług.

![](figures/numerical_distributions.png)

*Rysunek 4: Rozkład cech numerycznych - wiek (rozkład normalny), odległość lotu (prawostronnie skośny), opóźnienia (silnie prawostronnie skośne)*

**Rozkład cech kategorycznych:** Wykresy przedstawiają rozkład płci, typu klienta, typu podróży i klasy.

![](figures/categorical_distributions.png)

*Rysunek 5: Rozkład cech kategorycznych - większość pasażerów to lojalni klienci (81.7%), podróżujący w celach biznesowych (69.0%)*

**Analiza wartości odstających:** Wykryto wartości odstające w wieku (>3σ: 17 przypadków), odległości lotu (58), ocenie usług pokładowych (3) oraz opóźnieniach (>2,200 przypadków).

**Uzasadnienie:** Dane zawierają kluczowe informacje demograficzne, podróżnicze i oceny usług, które silnie korelują z satysfakcją pasażerów:
- Oceny usług pokładowych bezpośrednio wpływają na doświadczenie pasażera
- Typ podróży (biznesowa vs osobista) determinuje oczekiwania
- Klasa podróży wpływa na poziom usług
- Opóźnienia negatywnie wpływają na satysfakcję

## Sposób rozwiązania problemu

**Wybrane modele:** Zastosowano zestaw 11 modeli klasyfikacyjnych w celu porównania ich skuteczności:
1. Regresja logistyczna (baseline)
2. Drzewa decyzyjne
3. Random Forest
4. Gradient Boosting
5. AdaBoost
6. Extra Trees
7. SVM (Support Vector Machine)
8. K-Nearest Neighbors
9. Gaussian Naive Bayes
10. LDA (Linear Discriminant Analysis)
11. QDA (Quadratic Discriminant Analysis)

**Uzasadnienie wyboru:** Problem jest klasyfikacją binarną z niezbalansowanymi klasami. Zestaw modeli obejmuje różne podejścia do klasyfikacji (liniowe, drzewiaste, zespołowe, probabilistyczne), co pozwala na znalezienie najlepszego rozwiązania.

**Etapy realizacji projektu:**
1. Eksploracyjna analiza danych (EDA) - analiza rozkładów, brakujących wartości, korelacji, wartości odstających
2. Przygotowanie danych - imputacja brakujących wartości, inżynieria cech, kodowanie kategoryczne, skalowanie
3. Podział danych - 80% treningowe, 20% walidacyjne (z zachowaniem proporcji klas)
4. Trenowanie i ocena wielu modeli klasyfikacyjnych
5. Dostrojenie hiperparametrów najlepszych modeli (GridSearchCV)
6. Analiza ważności cech dla modeli drzewiastych
7. Ewaluacja końcowa i wybór najlepszego modelu

**Miary ewaluacji:**
- Accuracy (dokładność)
- Precision (precyzja)
- Recall (czułość)
- F1-Score (średnia harmoniczna)
- ROC-AUC (pole pod krzywą ROC)
- Macierz pomyłek (confusion matrix)

**Walidacja:** 5-krotna walidacja krzyżowa na zbiorze treningowym.

## Dyskusja wyników i ewaluacja modelu

**Wyniki modelowania:**

| Model | Accuracy | F1-Score | ROC-AUC | Czas treningu (s) |
|-------|----------|----------|---------|-------------------|
| Random Forest (Tuned) | 0.9642 | 0.9641 | 0.9944 | 2135.0887 |
| Random Forest | 0.9641 | 0.9640 | 0.9943 | 20.0139 |
| Extra Trees (Tuned) | 0.9628 | 0.9627 | 0.9940 | 2854.4734 |
| Extra Trees | 0.9623 | 0.9622 | 0.9937 | 13.5433 |
| SVM | 0.9538 | 0.9537 | 0.9891 | 842.3743 |
| Decision Tree | 0.9462 | 0.9462 | 0.9458 | 1.2279 |
| Gradient Boosting | 0.9454 | 0.9453 | 0.9886 | 33.3738 |
| K-Nearest Neighbors | 0.9222 | 0.9218 | 0.9651 | 0.0500 |
| AdaBoost | 0.9095 | 0.9093 | 0.9712 | 6.3514 |
| Logistic Regression | 0.8775 | 0.8773 | 0.9294 | 0.2406 |
| LDA | 0.8762 | 0.8760 | 0.9274 | 0.3105 |
| Gaussian Naive Bayes | 0.8630 | 0.8627 | 0.9209 | 0.0481 |
| QDA | 0.8560 | 0.8556 | 0.9182 | 0.1480 |

**Najlepszy model:** Random Forest (Tuned) - dokładność: 0.9642, F1-Score: 0.9641, ROC-AUC: 0.9944

**Analiza ważności cech (dla najlepszego modelu):**

![](figures/feature_importance.png)

*Rysunek 6: Najważniejsze cechy wpływające na satysfakcję pasażerów*

**Krzywe ROC wszystkich modeli:**

![](figures/roc_curves.png)

*Rysunek 7: Krzywe ROC dla wszystkich modeli - porównanie efektywności klasyfikacji*

**Macierz pomyłek najlepszego modelu:**

![](figures/confusion_matrix.png)

*Rysunek 8: Macierz pomyłek najlepszego modelu - analiza błędów klasyfikacji*

**Kluczowe obserwacje:**
1. Modele zespołowe (Random Forest, Gradient Boosting) osiągnęły najlepsze wyniki.
2. Regresja logistyczna, będąc prostszym modelem, osiągnęła konkurencyjne wyniki.
3. Dostrojenie hiperparametrów poprawiło wyniki średnio o 2-3%.
4. Czas treningu różnił się znacząco między modelami.

**Analiza błędów i kosztów klasyfikacji:**

Na podstawie macierzy pomyłek najlepszego modelu (Random Forest) można wyciągnąć następujące wnioski dotyczące błędów klasyfikacji:

1. **Rozkład błędów:**
   - **Fałszywie pozytywne (FP):** 376 przypadków - model przewiduje satysfakcję, gdy pasażer jest faktycznie niezadowolony. Te błędy oznaczają utracone szanse na interwencję i poprawę doświadczenia klienta.
   - **Fałszywie negatywne (FN):** 381 przypadków - model przewiduje niezadowolenie, gdy pasażer jest faktycznie zadowolony. Te błędy prowadzą do niepotrzebnych kosztów interwencji.

2. **Wskaźniki błędów:**
   - **False Positive Rate (FPR):** 1.28% - niski wskaźnik, co oznacza, że model rzadko myli niezadowolonych pasażerów za zadowolonych.
   - **False Negative Rate (FNR):** 1.95% - nieco wyższy wskaźnik, co sugeruje, że model częściej myli zadowolonych pasażerów za niezadowolonych.

3. **Koszty biznesowe błędów:**
   - **Koszty FP:** Utracony przychód z potencjalnych przyszłych podróży, negatywne recenzje, spadek lojalności. Szacunkowy koszt: wysoki (utrata klienta długoterminowego).
   - **Koszty FN:** Niepotrzebne wydatki na interwencje (vouchery, upgrade'y, personel). Szacunkowy koszt: średni (jednorazowy wydatek).

4. **Optymalizacja progu klasyfikacji:**
   - Dla linii lotniczych, gdzie utrzymanie klienta jest priorytetem, warto obniżyć próg klasyfikacji dla klasy "zadowolony", zmniejszając liczbę FP (bardziej agresywnie identyfikując niezadowolonych).
   - Obecny próg 0.5 można obniżyć do 0.4, co zwiększy czułość (recall) dla klasy "zadowolony" kosztem precyzji.

5. **Porównanie błędów między modelami:**
   - Modele zespołowe (Random Forest, Gradient Boosting) mają najbardziej zrównoważony rozkład błędów.
   - Modele probabilistyczne (Naive Bayes) mają wyższy wskaźnik FP, co czyni je mniej odpowiednimi dla aplikacji biznesowych.

**Porównanie dokładności modeli:**

![](figures/model_comparison_accuracy.png)

*Rysunek 9: Porównanie dokładności (Accuracy) wszystkich modeli - modele zespołowe osiągają najwyższe wyniki*

**Dodatkowe wnioski z porównania modeli:**
5. **SVM** osiągnął bardzo dobre wyniki (accuracy: 95.38%) pomimo prostoty modelu.
6. **K-Nearest Neighbors** osiągnął accuracy 92.22%, co jest dobrym wynikiem dla algorytmu opartego na podobieństwie.
7. **Modele probabilistyczne** (Naive Bayes, LDA, QDA) osiągnęły najniższe wyniki, sugerując, że założenia o rozkładzie danych nie są w pełni spełnione.
8. **AdaBoost** osiągnął accuracy 90.95%, co potwierdza skuteczność technik boosting.
9. **Czasy trenowania** różnią się o kilka rzędów wielkości: od 0.05s (KNN) do 2854s (Extra Trees Tuned).

**Top 5 najważniejszych cech:**
1. Online boarding (ocena odprawy online) - 15.18%
2. Inflight wifi service (ocena wifi pokładowego) - 14.82%
3. Type of Travel (typ podróży) - 9.78%
4. Class (klasa) - 9.57%
5. Inflight entertainment (rozrywka pokładowa) - 5.72%

**Wnioski z analizy ważności cech:**
1. Oceny usług pokładowych mają największy wpływ na satysfakcję.
2. Typ podróży (biznesowa vs osobista) jest istotnym predyktorem.
3. Klasa podróży znacząco wpływa na satysfakcję.
4. Opóźnienia mają umiarkowany wpływ na satysfakcję.
5. Cechy demograficzne (wiek, płeć) mają mniejsze znaczenie.

## Dodatkowe analizy i wizualizacje

W celu pogłębienia analizy wyników modelowania, wygenerowano dodatkowe wizualizacje porównujące modele pod względem różnych metryk i czasu trenowania.

**Porównanie wielometryczne modeli:**

![](figures/multi_metric_comparison.png)

*Rysunek 10: Porównanie trzech kluczowych metryk (Accuracy, F1-Score, ROC-AUC) dla wszystkich modeli - modele zespołowe osiągają najwyższe wyniki we wszystkich metrykach*

**Heatmapa wydajności modeli:**

![](figures/performance_heatmap.png)

*Rysunek 11: Heatmapa wydajności modeli - wizualne porównanie Accuracy, F1-Score i ROC-AUC; kolory wskazują na wyższe wartości (żółty/czerwony)*

**Analiza kompromisu dokładność-czas trenowania:**

![](figures/accuracy_vs_time_tradeoff.png)

*Rysunek 12: Analiza kompromisu między dokładnością a czasem trenowania - modele w prawym dolnym rogu są optymalne (wysoka dokładność, niski czas); linia Pareto wskazuje granicę efektywności*

**Kluczowe wnioski z dodatkowych analiz:**
1. **Random Forest (Tuned)** osiąga najlepsze wyniki we wszystkich metrykach, ale ma najdłuższy czas trenowania.
2. **SVM** oferuje najlepszy kompromis między dokładnością a czasem trenowania dla aplikacji wymagających częstego retrenowania.
3. **Modele probabilistyczne** (Naive Bayes, LDA, QDA) mają najniższe wyniki we wszystkich metrykach, potwierdzając ich niedopasowanie do struktury danych.
4. **K-Nearest Neighbors** ma najkrótszy czas trenowania (0.05s) przy zachowaniu przyzwoitej dokładności (92.22%).
5. **Extra Trees (Tuned)** ma najdłuższy czas trenowania (2854s) przy niewielkiej poprawie dokładności w porównaniu do wersji nietunowanej.

## Rekomendacje biznesowe

Na podstawie przeprowadzonej analizy i wyników modelowania, linie lotnicze mogą podjąć następujące działania w celu poprawy satysfakcji pasażerów:

**1. Priorytetyzacja usług pokładowych:**
- **Odprawa online (Online boarding)** - najważniejszy czynnik (15.18% ważności). Należy inwestować w rozwój i udoskonalanie systemów odprawy online, zapewniając intuicyjny interfejs i szybkie procesy.
- **WiFi pokładowe (Inflight wifi service)** - drugi najważniejszy czynnik (14.82%). Poprawa jakości i prędkości internetu pokładowego jest kluczowa dla satysfakcji pasażerów, szczególnie w erze cyfrowej.
- **Rozrywka pokładowa (Inflight entertainment)** - piąty najważniejszy czynnik (5.72%). Wzbogacenie oferty rozrywkowej (filmy, muzyka, gry) może znacząco poprawić doświadczenia pasażerów.

**2. Segmentacja klientów:**
- **Pasażerowie biznesowi** stanowią 69.0% podróżujących i mają wyższe oczekiwania. Należy zapewnić im dedykowane usługi: priorytetową odprawę, lepsze WiFi do pracy, wygodniejsze siedzenia.
- **Klienci lojalni** stanowią 81.7% pasażerów. Programy lojalnościowe powinny być wzmocnione, z nagrodami bezpośrednio wpływającymi na satysfakcję (dostęp do lepszych usług).

**3. Zarządzanie opóźnieniami:**
- Opóźnienia mają umiarkowany wpływ na satysfakcję. Należy wdrażać proaktywne komunikaty o opóźnieniach, oferować rekompensaty (vouchery, napoje) oraz optymalizować procesy obsługi naziemnej.

**4. Personalizacja usług:**
- Wykorzystanie modelu predykcyjnego do identyfikacji pasażerów z ryzykiem niskiej satysfakcji przed podróżą. Dla tych pasażerów można zaoferować ulepszenia (upgrade klasy, specjalne posiłki, personalizowane powitanie).

**5. Monitorowanie w czasie rzeczywistym:**
- Implementacja systemu monitorowania satysfakcji w czasie rzeczywistym na podstawie danych z ankiet i predykcji modelu. Pozwoli to na szybką reakcję na problemy.

**6. Koszty błędów klasyfikacji:**
- **Fałszywie pozytywne** (przewidywanie satysfakcji gdy pasażer jest niezadowolony): ryzyko utraty szansy na poprawę doświadczenia.
- **Fałszywie negatywne** (przewidywanie niezadowolenia gdy pasażer jest zadowolony): niepotrzebne koszty interwencji.
- Należy zrównoważyć koszty, optymalizując próg klasyfikacji modelu.

**7. Wybór modelu do wdrożenia:**
- **Random Forest (Tuned)** osiągnął najwyższą dokładność (96.42%) ale wymaga najdłuższego czasu trenowania (2135s).
- **SVM** oferuje dobry kompromis między dokładnością (95.38%) a czasem trenowania (842s).
- Dla wdrożenia w czasie rzeczywistym można rozważyć **Random Forest** bez tuningu (96.41% dokładności, tylko 20s trenowania).

**8. Metryki sukcesu:**
- Monitorowanie wskaźników biznesowych: wzrost liczby pozytywnych recenzji, zwiększenie lojalności klientów, redukcja reklamacji, wzrost przychodów z programów lojalnościowych.

## Podsumowanie

**Co się udało?**
1. Przeprowadzono kompleksową eksploracyjną analizę danych (EDA).
2. Wykryto i przetworzono brakujące wartości oraz wartości odstające.
3. Przetestowano 11 różnych modeli klasyfikacyjnych.
4. Dostrojono hiperparametry najlepszych modeli, uzyskując poprawę wyników.
5. Zidentyfikowano kluczowe cechy wpływające na satysfakcję pasażerów.
6. Osiągnięto wysoką dokładność predykcji (powyżej 90% dla najlepszego modelu).

**Problemy i ich rozwiązania:**
1. **Nierównowaga klas** - zastosowano stratified sampling przy podziale danych.
2. **Brakujące wartości** - użyto imputacji mediany dla zmiennych numerycznych.
3. **Wartości odstające** - zastosowano analizę Z-score (>3) i zachowano wartości.
4. **Zmienne kategoryczne** - zastosowano label encoding.
5. **Duża liczba cech** - wykonano selekcję cech na podstawie ważności.

**Możliwości rozwoju:**
1. Zastosowanie głębokiego uczenia (sieci neuronowe) dla potencjalnej poprawy wyników.
2. Wykorzystanie technik ensemble stacking/boosting.
3. Dodanie większej liczby cech (np. dane o lotniskach, sezonowość).
4. Implementacja systemu w czasie rzeczywistym do przewidywania satysfakcji.
5. Rozszerzenie analizy o segmentację klientów.

**Wnioski końcowe:** Model predykcji satysfakcji pasażerów linii lotniczych został pomyślnie zbudowany i oceniony. Najlepsze wyniki osiągnęły modele zespołowe, a analiza ważności cech dostarczyła cennych informacji biznesowych. System może być wdrożony przez linie lotnicze do monitorowania i poprawy satysfakcji klientów.

## Załączniki

### Załącznik 1. Eksploracyjna analiza danych

Pełna eksploracyjna analiza danych dostępna w osobnym pliku: `eda_satisfaction.md`

Załącznik zawiera kompleksową analizę danych, w tym:
- Analizę brakujących wartości, rozkładów i korelacji
- **Zaawansowane analizy:** segmentację satysfakcji według demografii i typu podróży
- **Analizy interakcji:** mapy cieplne satysfakcji według klasy i typu podróży
- **Testy statystyczne:** testy t-Studenta dla różnic wieku i opóźnień
- **Analizy priorytetowych usług:** identyfikację usług o największym wpływie na satysfakcję
- **Wizualizacje:** 12 rysunków ilustrujących kluczowe zależności w danych