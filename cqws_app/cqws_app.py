import os
import sentry_sdk
import streamlit as st

if os.path.isfile(".streamlit/secrets.toml"):
    if 'sentry_url' in st.secrets:
        sentry_sdk.init(
            st.secrets['sentry_url'],
            # Set traces_sample_rate to 1.0 to capture 100%
            # of transactions for performance monitoring.
            # We recommend adjusting this value in production.
            traces_sample_rate=0.001,
        )
    else:
        print('sentry not running')
else:
    print('Ok!')
    
    