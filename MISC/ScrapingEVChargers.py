# %%
import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import numpy as np
import time
import math

#%%

def get_enbw(min_lat, max_lat, min_lon, max_lon):
    headers = {
        'Connection': 'keep-alive',
        'Accept': 'application/json',
        'Accept-Language': 'de',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.121 Safari/537.36',
        'Ocp-Apim-Subscription-Key': 'bd155d66715f4629af837c20ce31377f',
        'Origin': 'https://www.enbw.com',
        'Sec-Fetch-Site': 'same-site',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Dest': 'empty',
        'Referer': 'https://www.enbw.com/',
    }

    params = (
        ('fromLat', min_lat),
        ('toLat', max_lat),
        ('fromLon', min_lon),
        ('toLon', max_lon),
        ('grouping', 'false'),
    )

    response = requests.get('https://api.enbw.com/emobility-public/api/v1/chargestations', headers=headers, params=params)
    enbw = pd.DataFrame(json.loads(response.content)).drop(['viewPort', 'distanceToMyPositionInKm', 'favorite', 'hiddenWithActiveProfile'], axis=1)

    return enbw

#%%
enbw = get_enbw(
    min_lat='52.304188378859486',
    max_lat='52.53367617203326',
    min_lon='12.66139243769531',
    max_lon='13.330185162304685'
    )

# %%


def _get_plugsurfing_id_list(min_lat, max_lat, min_lon, max_lon):
    headers = {
        'Connection': 'keep-alive',
        'Accept': 'application/json, text/plain, */*',
        'Authorization': '88234shdfsdkl0_$1sdvRd01_233fdd',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.121 Safari/537.36',
        'Origin': 'https://www.plugsurfing.com',
        'Sec-Fetch-Site': 'same-site',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Dest': 'empty',
        'Referer': 'https://www.plugsurfing.com/',
        'Accept-Language': 'en-GB,en-US;q=0.9,en;q=0.8,de;q=0.7',
    }

    params = (
        ('min-lat', min_lat),
        ('max-lat', max_lat),
        ('min-lon', min_lon),
        ('max-lon', max_lon),
    )

    response = requests.get('https://api.plugsurfing.com/persik/map-charging-stations', headers=headers, params=params)

    return pd.DataFrame(json.loads(response.content))

def get_plugsurfing(min_lat, max_lat, min_lon, max_lon):
    data = _get_plugsurfing_id_list(min_lat, max_lat, min_lon, max_lon)

    n_iter = math.ceil(len(data)/30)
    results = []

    for i in range(n_iter):
        print(f"[FETCHING] Iteration {i+1}/{n_iter}...")
        time.sleep(1)
        from_idx = i * 30
        to_idx = i * 30 + 30
        
        if i == n_iter-1:
            to_idx = len(data) - 1

        ids_to_fetch = ','.join(''.join([char for char in np.array2string(data.id.values) if char not in ['\n', '[', ']']]).split()[from_idx: to_idx])

        params = (
            ('ids', ids_to_fetch),
        )

        headers = {
            'Connection': 'keep-alive',
            'Accept': 'application/json, text/plain, */*',
            'Authorization': '88234shdfsdkl0_$1sdvRd01_233fdd',
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.121 Safari/537.36',
            'Origin': 'https://www.plugsurfing.com',
            'Sec-Fetch-Site': 'same-site',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Dest': 'empty',
            'Referer': 'https://www.plugsurfing.com/',
            'Accept-Language': 'en-GB,en-US;q=0.9,en;q=0.8,de;q=0.7',
        }

        response = requests.get('https://api.plugsurfing.com/persik/charging-stations-by-ids', headers=headers, params=params)

        if not response.ok:
            print(f"[FAILED]: {response.content}")

        fetch_results = pd.DataFrame(json.loads(response.content)['stations'])

        results.append(fetch_results)
        print(f"--> Successfully appended {len(fetch_results)} results!")

    return pd.concat(results, ignore_index=True)

#%%
plugsurfing = get_plugsurfing(
    min_lat='52.304188378859486',
    max_lat='52.53367617203326',
    min_lon='12.66139243769531',
    max_lon='13.330185162304685'
    )

#%%
plugsurfing[['latitude', 'longitude']] = plugsurfing[['latitude', 'longitude']].round(3)
enbw[['lat', 'lon']] = enbw[['lat', 'lon']].round(3)

stations = pd.merge(enbw, plugsurfing, left_on=['lat', 'lon'], right_on=['latitude', 'longitude'])

#%%
stations['price_data'] = stations.connectors.apply(lambda r: r[0]['tariff']['elements'][0]['pricesComponents'])
stations['pricing'] = stations.price_data.apply(lambda r: [item['price'] for item in r])