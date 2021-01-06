# OpenCV - Automobile
Raspberry Pi를 하드웨어의 기반으로 둔, openCV & Python 자율 주행 프로젝트 

# What is that?
일단 이게 원래 시연회를 위해서 쓰인 프로젝트 였으나 이게 어쩌저찌 하다가 터졌다. 자율주행에 대한 자료와 코드를 긁어 오기 위해서 진행한 프로젝트 였다. 개인 시연회 진행 예정 이였으나 아마 이게 생각보다 안 잡히는 문제도 많고 실질적인 맵 만들기를 혼자서 진행하기 어려워서 였나 암튼 여러 이유로 취소 되었다.

# 아니 일단 뭔지는 설명해야지
음 ㅇㅋ. 솔직히 거의 오픈 소스 집합체 이긴 한데 perspective.py 같은 경우는 실질적으로 가장 많이 테스트를 진행한 코드로 기억을 하며 perspective_Image.py 는 말 그대로 image 파일을 읽어와서 테스트를 진행한다. perspective_Video.py 는 video 파일을 읽어와서 테스트를 한다. 참고로 perspective_Video.py 에서 <a href="https://docs.opencv.org/3.4/d8/dfe/classcv_1_1VideoCapture.html">cv2.VideoCapture()</a> 에서 값을 0으로 넣어주게 되면 실시간 처리가 가능하다. <br> <br>

## ./openCV-automobile/line_images
안에 있는 line.py 는 테스트를 진행 할때 가장 빠르게 lane을 잡는 코드 이다. 내 기억상 real-time이 되었던거 같은데 lane은 잡지만 그에 대한 다른 액션을 취하진 못한다. 그닌까 왼쪽 오른쪽 구분을 할줄은 모른다 이런말이지.

# Data
사진이랑 영상들은 직접 하나하나 뒤져가면서 찾은 것 들이다. 누군가에게는 이거라도 도움이 되길 바라며..
```bash
./openCV-automobile/videos
./openCV-automobile/images
```
