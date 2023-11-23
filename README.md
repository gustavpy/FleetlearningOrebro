Fleet Learning

Vi fyra som går Young Talent programmet på AI Sweden vid Örebro Universitet (Arvid, Gustav, Joel och Valter) har varit i ett projekt med Zenseact som gör mjukvaran till Volvos självkörande bilar. I mjukvaran tränar modellen upp sig på bilder. Vår uppgift har varit bland annat att sammanställa data, jämföra olika strategier och undersöka vad som kan påverka träningen negativt.

Vi har fått tillgång till ZOD (Zenseact open dataset) som är ett dataset med 100 000 bilder och tillhörande metadata. Vi har även fått tillgång till koden som tränar upp en modell på bilderna som kan välja vart bilen ska åka i varje situation. 

Vi började med att skapa en DataFrame för metadatan till alla bilderna i datasetet och det fanns följande värden i metadatan:

"""frame_id,time,country_code,scraped_weather,collection_car,road_type,road_condition,time_of_day,num_lane_instances,num_vehicles,num_vulnerable_vehicles,num_pedestrians,num_traffic_lights,num_traffic_signs,longitude,latitude,solar_angle_elevation"""
