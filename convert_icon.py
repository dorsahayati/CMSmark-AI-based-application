from PIL import Image

img = Image.open('icon/app_icon.JPG')
# ICO files typically use sizes: 16x16, 32x32, 48x48, 256x256
img.save('icon/app_icon.ico', format='ICO', sizes=[(16,16), (32,32), (48,48), (256,256)])
print("Icon converted successfully!")