import textdistance


a = "Panasonic VIERA 50' Plasma Flat Panel 1080p HDTV In Black - TH50PZ85U".lower()
b = "Panasonic Viera TH-50PZ85U 50' Plasma TV".lower()
d = "Panasonic VIERA TH-50PX80U 50' Plasma TV".lower()
c = textdistance.overlap.normalized_similarity(a, b)
e = textdistance.overlap.normalized_similarity(a, d)
print(c)
print(e)