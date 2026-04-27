class AspectRatioProcessor:
    """
    Kelas untuk menganalisis rasio aspek dari sekumpulan landmarks.
    """
    def __init__(self, image_width: int, image_height: int):
        self.w = image_width
        self.h = image_height

    def get_aspect_ratio(self, landmarks):
        """
        Menghitung rasio tinggi terhadap lebar dari landmarks.
        """
        # Konversi koordinat relatif ke piksel
        x_coords = [lm.x * self.w for lm in landmarks]
        y_coords = [lm.y * self.h for lm in landmarks]
        
        # Hitung lebar dan tinggi asli
        width = max(x_coords) - min(x_coords)
        height = max(y_coords) - min(y_coords)
        
        # Mencegah error pembagian dengan nol
        if width == 0:
            return 0.0
            
        return height / width