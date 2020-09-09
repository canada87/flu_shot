import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import streamlit as st

class preprocess_data:

    def eda(self, data):
        # st.write(data.isna().sum())

        global_dict = dict()
        cols_name = ['age_group', 'education', 'race', 'sex', 'income_poverty', 'marital_status', 'rent_or_own',
                    'employment_status', 'hhs_geo_region', 'census_msa', 'employment_industry',
                    'employment_occupation']

        for col in cols_name:
            data[col] = data[col].fillna('unknown')
            le = LabelEncoder()
            data[col]= le.fit_transform(data[col])
            dict_class = dict(zip(le.classes_, range(len(le.classes_))))
            global_dict[col] = dict_class

        for col in data.columns:
            data[col] = data[col].fillna(len(data[col].unique())-1)

        # st.write(data)
        # st.write(global_dict)
        return data, global_dict
