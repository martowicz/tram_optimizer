import os
import pandas as pd
from weasyprint import HTML

def generate_pdf_posters(csv_filepath: str, stop_names_dictionary: dict, output_dir: str = "data/output"):
    """
    Na podstawie gotowego rozkładu jazdy z pliku CSV oraz słownika nazw przystanków,
    generuje i zapisuje plakaty PDF dla każdego przystanku z osobna.
    """
    if not os.path.exists(csv_filepath):
        print(f"❌ Błąd: Nie znaleziono pliku {csv_filepath}")
        return
        
    print(f"📖 Wczytywanie rozkładu jazdy z pliku: {csv_filepath}...")
    df_schedule = pd.read_csv(csv_filepath)
    
    os.makedirs(output_dir, exist_ok=True)
    
    unikalne_przystanki = sorted(df_schedule["Stop"].unique())
    
    ostatni_przystanek_id = max(stop_names_dictionary.keys())
    kierunek = stop_names_dictionary.get(ostatni_przystanek_id, "Koniec trasy")
    
    for stop_id in unikalne_przystanki:
        stop_name = stop_names_dictionary.get(stop_id, f"Przystanek {stop_id}")
        
        df_stop = df_schedule[df_schedule["Stop"] == stop_id].copy()
        
        rozklad_godzinowy = {h: [] for h in range(24)}
        
        for _, row in df_stop.iterrows():
            if pd.isna(row["ClockTime"]):
                continue

            hh, mm = map(int, str(row["ClockTime"]).split(":"))
            rozklad_godzinowy[hh].append(f"{mm:02d}")
            

        trasa_html = " &rarr; ".join([
            f"<strong>{stop_names_dictionary.get(k, str(k))}</strong>" if k == stop_id else stop_names_dictionary.get(k, str(k))
            for k in sorted(stop_names_dictionary.keys())
        ])
        
        wiersze_html = ""
        for h in range(24):
            minuty = sorted(rozklad_godzinowy[h])
            minuty_str = " ".join([f"<span>{m}</span>" for m in minuty]) if minuty else "&mdash;"
            
            wiersze_html += f"""
            <tr style="border-bottom: 1px solid #e2e8f0;">
                <td style="background-color: #2c3e50; color: white; font-weight: bold; font-size: 14pt; width: 60px; text-align: center; padding: 10px 0;">{h:02d}</td>
                <td style="padding-left: 20px; font-size: 13pt; letter-spacing: 3px; color: #333333;">{minuty_str}</td>
            </tr>
            """
            
        szablon_html = f"""
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                @page {{ size: A4; margin: 15mm 12mm; }}
                body {{ font-family: Arial, sans-serif; color: #2c3e50; margin: 0; }}
                .header {{ background-color: #1a3a5f; color: white; padding: 20px; border-radius: 4px; }}
                .line-badge {{ display: inline-block; background-color: #e74c3c; color: white; font-size: 28pt; font-weight: bold; padding: 5px 20px; border-radius: 4px; margin-right: 15px; vertical-align: middle; }}
                .title-box {{ display: inline-block; vertical-align: middle; }}
                .route-box {{ margin: 20px 0; background-color: #f8f9fa; padding: 12px; border-left: 5px solid #1a3a5f; font-size: 10pt; }}
            </style>
        </head>
        <body>
            <div class="header">
                <div class="line-badge">50</div>
                <div class="title-box">
                    <h1 style="margin: 0; font-size: 20pt;">{stop_name}</h1>
                    <p style="margin: 5px 0 0 0; color: #bdc3c7;">Kierunek: {kierunek}</p>
                </div>
            </div>
            
            <div class="route-box">
                <strong>Trasa:</strong> {trasa_html}
            </div>
            
            <table style="width: 100%; border-collapse: collapse;">
                <thead>
                    <tr style="background-color: #34495e; color: white;">
                        <th style="padding: 8px; text-align: center;">Godz.</th>
                        <th style="padding: 8px; text-align: left; padding-left: 20px;">Minuty odjazdów</th>
                    </tr>
                </thead>
                <tbody>
                    {wiersze_html}
                </tbody>
            </table>
            
            <p style="text-align: center; margin-top: 30px; font-size: 8pt; color: #7f8c8d; border-top: 1px dashed #bdc3c7; padding-top: 10px;">
                Rozkład jazdy wygenerowany automatycznie systemem optymalizacji.
            </p>
        </body>
        </html>
        """
        
        temp_html_path = f"temp_stop_{stop_id}.html"
        pdf_output_path = f"{output_dir}/rozkład_przystanek_{stop_id}.pdf"
        
        with open(temp_html_path, "w", encoding="utf-8") as f:
            f.write(szablon_html)
            
        HTML(temp_html_path).write_pdf(pdf_output_path)
        os.remove(temp_html_path)
        
    print(f"✅ Sukces! Wygenerowano {len(unikalne_przystanki)} plakatów PDF w folderze {output_dir}/")