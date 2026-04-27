class BoundingBoxProcessor:
    """
    Kelas untuk memproses perhitungan Bounding Box dari landmarks.
    """
    def __init__(self, image_width: int, image_height: int):
        self.w = image_width
        self.h = image_height

    def get_padded_bbox(self, landmarks, padding: int):
        """
        Menghitung bounding box dengan tambahan padding.
        """
        # Konversi koordinat relatif ke piksel
        x_coords = [int(lm.x * self.w) for lm in landmarks]
        y_coords = [int(lm.y * self.h) for lm in landmarks]

        # Cari titik minimum dan maksimum
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        # Tambahkan padding dan pastikan tidak keluar dari batas gambar
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(self.w, x_max + padding)
        y_max = min(self.h, y_max + padding)

        return x_min, y_min, x_max, y_max