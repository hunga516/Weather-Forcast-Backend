from meteostat import Point, Hourly
import pandas as pd
from datetime import datetime

import ssl

# Tạo SSL context không xác thực chứng chỉ
ssl._create_default_https_context = ssl._create_unverified_context

# Xác định khoảng thời gian muốn lấy dữ liệu (2009 - 2015)
start = datetime(2009, 1, 1)
end = datetime(2015, 12, 31)

# Tọa độ của Quận 1, Quận 2, Quận 3 ở TPHCM
districts = {
    'Quận 1': {'lat': 10.7758, 'lon': 106.7009},
    # 'Quận 2': {'lat': 10.7872, 'lon': 106.7496},
    # 'Quận 3': {'lat': 10.7794, 'lon': 106.6843},
}

# Tạo một DataFrame để lưu dữ liệu thời tiết của từng quận
weather_data = pd.DataFrame()

# Lặp qua từng quận và lấy dữ liệu thời tiết
for district, coords in districts.items():
    # Khởi tạo vị trí dựa trên tọa độ của từng quận
    location = Point(coords['lat'], coords['lon'])
    
    # Lấy dữ liệu thời tiết theo giờ
    data = Hourly(location, start, end)
    data = data.fetch()
    
    # Chỉ giữ lại các giờ cách nhau 3 giờ (0, 3, 6, 9, 12, 15, 18, 21)
    data = data[data.index.hour % 3 == 0]
    
    # Thêm tên quận vào DataFrame để dễ nhận diện
    data['District'] = district
    
    # Gộp dữ liệu của từng quận vào DataFrame chung
    weather_data = pd.concat([weather_data, data])

# Reset lại index để dữ liệu trông dễ nhìn hơn
weather_data.reset_index(inplace=True)

# Lưu dữ liệu vào file CSV
weather_data.to_csv('weather_data_2009_2015_QUAN1.csv', index=False, encoding='utf-8')

# Hiển thị thông báo hoàn tất
print("Dữ liệu đã được lưu vào file 'weather_data_2009_2015_QUAN1.csv'")
