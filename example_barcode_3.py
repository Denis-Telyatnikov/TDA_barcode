import TDA_barcode.barcode as bc

# ----------------------------------------------------------
# Пример визуализации и сравнения баркодов H0 - H4.
# ----------------------------------------------------------

barcode1 = [
    [[0, 'infinity'], [0, 4], [0, 4], [0, 5], [0, 6], [0, 11], [0, 5], [0, 3], [0, 4], [0, 4], [0, 5], [0, 8], [0, 9]],
    [[2, 4.5], [2, 3], [2, 2.5], [2.5, 4], [2.5, 6], [2.5, 4.5], [3, 6], [4, 10], [4.5, 10.5], [6, 10], [7.5, 10]],
    [[1, 7], [1.5, 9], [2, 5.5], [2, 3], [2, 2.5], [2, 5], [3, 7], [3.5, 6], [3.5, 4.5], [5, 8], [5, 9]],
    [[1, 2], [1, 4], [1, 1.5], [2, 8], [2, 4], [3, 4], [3, 6], [3, 3.5], [4, 9], [4, 7], [4, 5]],
    [[2, 5], [3, 5], [4, 7], [4, 9]]
]

barcode2 = [
    [[0, 'infinity'], [0, 5], [0, 2], [0, 5], [0, 7], [0, 9], [0, 8], [0, 8], [0, 2]],
    [[1, 6], [2, 4], [2.5, 6.6], [3, 5], [3, 4], [5, 10], [5, 7], [5, 9], [8, 9]],
    [[1, 2], [1, 6], [1, 3], [2, 9], [2.5, 6], [2.5, 7], [3, 6], [3, 8], [4, 9], [4, 6], [4, 5]],
    [[4, 5], [4, 4.5], [4.5, 5.5], [6, 7.5]],
    [[1, 3], [1, 2], [1, 8], [2, 3], [2.5, 7], [3, 7], [4, 6], [4, 5.5], [4, 5], [5, 8.5]]
]


bc.visualize_barcode(barcode1)
bc.visualize_barcode(barcode2)

insignificant = 1
bc.two_barcodes_on_diagram(barcode1, barcode2)
bc.compare_two_barcodes(barcode1, barcode2, insignificant)
