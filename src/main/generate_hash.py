import streamlit_authenticator as stauth

password = "123456"   # change these

hasher = stauth.Hasher()
hashed_password = hasher.hash(password)

print(hashed_password)
