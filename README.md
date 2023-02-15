# neural_network_handling_code_pytorch
 뉴럴네트워크 구조에 따른 연산을 pytorch를 활용해서 조작하는 방법을 공부해보는 레포


## Autoencoder를 활용한 latent vector space의 feature 추출
   - autoencoder_network 폴더
     1) MNIST 데이터로 Autoencoder network를 학습하여 Reconstruction 모델을 구축.
     2) T-SNE 차원 축소 기법을 활용하여 2, 3 차원으로 축소 후 test data label과 매칭하여 클러스터링 분석 및 시각화.
       
       -> 실험 목적: latent vector의 차원을 축소 했을 때, 데이터의 군집화가 잘 되는 지 파악하기 위함.
     3) Validation data에서 Target Class 1개에 매칭되는 input, label 데이터를 활용하여 latent vector를 얻고, 얻은 latent vector와 test data에서 얻은 latent vector들과의 코사인 유사도를 산정하고, test data Reconstruction 추론 데이터를 유사도가 높은 기준으로 정렬하여 시각화 분석을 수행.
       
       -> 실험 목적: target class에 코사인 유사도가 높은 데이터가 생성되는지 확인할 수 있음.
