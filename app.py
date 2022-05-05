
#streamlit 라이브러리를 불러오기
import streamlit as st
#AI모델을 불러오기 위한 joblib 불러오기
import joblib
import pandas as pd

# st를 이용하여 타이틀과 입력 방법을 명시한다.

def user_input_features() :
  signalstrength = st.sidebar.number_input("신호강도: ")
  antenacoverage =st.sidebar.number_input("안테나 커버리지: ") 
  antenalength = st.sidebar.number_input("안테나길이: ")
  bandwidth =st.sidebar.number_input("밴드범위: ")

  data = {'signalstrength' : [signalstrength],
          'antenacoverage' : [antenacoverage],
          'antenalength' : [antenalength],
          'bandwidth' : [bandwidth],
          }
  data_df = pd.DataFrame(data, index=[0])
  return data_df

# new_x = {"signalstrenth" : [4.0],"antenacoverage" : [3.0],"antenalength": [5.0], "bandwidth": [2.0] }


st.title('민원의 내용을 3가지 VOC로 분류하세요')
st.markdown('* 우측에 데이터를 입력해주세요')

le_call = joblib.load("le.pkl")
scaler_call = joblib.load("scaler.pkl")
model_call = joblib.load("model.pkl")


new_x_df = user_input_features()

data_con_scale = scaler_call.transform(new_x_df)
result = model_call.predict(data_con_scale) 
le_call.inverse_transform(result)


#예측결과를 화면에 뿌려준다. 
st.subheader('결과는 다음과 같습니다.')
st.write('예상 민원 등급:', result[0])