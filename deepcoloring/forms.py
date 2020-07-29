from django import forms

class ImageForm(forms.Form):
  image = forms.ImageField(
    error_messages={
      'missing': '이미지 파일이 선택되지 않았습니다.',
      'invalid': '분류할 이미지 파일을 선택해 주세요.',
      'invalid_image': '이미지 파일이 아닙니다.'})

