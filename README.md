# Deepcolorization

- 모델 다운로드:
    * 해당 프로젝트를 실행시키기 위한 모델을 다운로드 받습니다.
    ```
    mkdir -p ./checkpoints/siggraph_retrained
    MODEL_FILE=./checkpoints/siggraph_retrained/latest_net_G.pth
    URL=http://colorization.eecs.berkeley.edu/siggraph/models/pytorch.pth
    wget -N $URL -O $MODEL_FILE
    ```
    * ./checkpoints/siggraph_retrained/latest_net_G.pth 에 맞게 경로를 설정해주세요.
    
- 프로젝트 실행:
    ``` python manage.py runserver ```
    를 입력해 실행합니다.


# FilterRemover

![dcz_rf](https://user-images.githubusercontent.com/29967386/84888346-1cbd2500-b0d2-11ea-9105-5416762aa25d.gif)

- 필터가 씌워진 사진을 첨부합니다.

![unnamed](https://user-images.githubusercontent.com/29967386/84734705-983caa80-afdc-11ea-9138-47e4faa16745.jpg)
![unnamed (2)](https://user-images.githubusercontent.com/29967386/84734715-9d99f500-afdc-11ea-8da3-5393b4c62f45.jpg)
![unnamed (3)](https://user-images.githubusercontent.com/29967386/84734724-a1c61280-afdc-11ea-8a00-be5942ca0be4.jpg)

- 위 이미지들과 같이 필터가 씌워진 사진을 삽입하면, 필터가 제거된 이미지가 출력됩니다.

![00000000_0p031_real](https://user-images.githubusercontent.com/29967386/84729526-df23a380-afce-11ea-8093-42a74fd5528b.png)

# Colorization

![dcz_c-_1_](https://user-images.githubusercontent.com/29967386/84888378-2ba3d780-b0d2-11ea-9908-ca4854607103.gif)

- 해당 이미지와 같이 256x256 크기(권장)의 이미지를 input data로 넣습니다. (256x256 크기가 아니어도 되지만 resize 과정에서 픽셀이 깨지는 현상이 발생합니다)

![00000000_0p031_real](https://user-images.githubusercontent.com/29967386/84729526-df23a380-afce-11ea-8093-42a74fd5528b.png)

- 해당 이미지에서 특정 부분의 색상값을 가져와 해당 픽셀부분에 적용시킵니다.

![00000000_0p031_hint_ab](https://user-images.githubusercontent.com/29967386/84729446-af749b80-afce-11ea-9541-126a7bfbfa8b.png)
![00000000_0p031_fake_reg](https://user-images.githubusercontent.com/29967386/84729470-be5b4e00-afce-11ea-8fae-bd77dd6de507.png)

- 이 기술을 토대로, 사용자는 특정 부분에 원하는 색상값을 주입시킬 수 있습니다. 이를 통해 해당 부분에 새로운 색깔을 추가하는 것이 가능합니다

![00000001_0p031_hint_ab](https://user-images.githubusercontent.com/29967386/84729585-0ed2ab80-afcf-11ea-8586-ddcd71f8beb0.png)
![00000001_0p031_fake_reg](https://user-images.githubusercontent.com/29967386/84729588-109c6f00-afcf-11ea-9a0b-186112dd2484.png)

# Django Project

- 실행 1. 메인화면

![image](https://user-images.githubusercontent.com/29967386/84734914-27e25900-afdd-11ea-864f-b32b825e06c6.png)

   * 위 버튼 중 1번(왼쪽, Remove Photo Filter)을 선택하면 사진의 필터를 지우는 작업을 수행할 수 있습니다.
   * 위 버튼 중 2번(오른쪽, Colorization)을 선택하면 사진에 새로운 색상을 입힐 수 있습니다.
   
- 실행 2. 필터 제거

![image](https://user-images.githubusercontent.com/29967386/84735055-7a237a00-afdd-11ea-9e7d-872e5eb0c34d.png)

   * 해당 화면에서 중앙의 Choose File 버튼을 클릭해 이미지를 첨부하고 완료하기 버튼을 눌러 필터를 제거할 수 있습니다.
   
   ![image](https://user-images.githubusercontent.com/29967386/84735202-d9818a00-afdd-11ea-992a-e4ba40e5c9d6.png)
   
   * 필터가 제거된 사진. 사진을 클릭해 다운로드 하거나 이어서 채색하기 버튼을 클릭해 Colorization을 할 수 있습니다.

- 실행 3. 채색하기

![image](https://user-images.githubusercontent.com/29967386/84735318-2f563200-afde-11ea-9bae-70db2e712ccd.png)

   * 왼쪽의 Choose File 버튼을 클릭해 파일을 첨부하고, 이미지에 삽입할 색상 값을 오른쪽 상단의 색상 바를 통해 조절합니다.
   * 색상 값을 변경한 후 오른쪽 이미지를 클릭하면 해당 이미지에 해당 색상값을 추가시킬 수 있습니다.
   * 왼쪽 하단의 버튼 중 되돌리기 버튼을 통해 마지막으로 추가한 색상을 제거할 수도 있습니다.
   * 마지막으로 완료하기 버튼을 클릭해 이미지를 Colorization 합니다.
   
   ![image](https://user-images.githubusercontent.com/29967386/84735566-cde29300-afde-11ea-8903-6f6728ec3afc.png)
   
   * 추가된 색상값을 반영해 랜덤하게 만들어진 10장의 이미지가 출력됩니다. 이전과 마찬가지로 클릭해 다운로드 하거나, 다시 한번 Colorization을 수행할 수도 있습니다.
