
import pandas as pd
import os
from urllib.parse import urlparse

df = pd.read_csv("2_testa_user.csv")
print(df)


final_dict = {}

for i,val in enumerate(df['image_url']) :
    # print(i,val)

    url = urlparse(val)
    fn = os.path.basename(url.path)
    fn_noext = os.path.splitext(fn)[0]
    print(fn_noext)
    final_dict [i] = fn_noext




final_dict_fn_id = {y:x for x,y in final_dict.items()}

print(final_dict_fn_id)


import json
with open('link_image_fn_id.json', 'w') as fp:
    json.dump(final_dict_fn_id, fp)









