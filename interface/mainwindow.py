import sys
from PyQt5.QtWidgets import QApplication, QPushButton

app = QApplication(sys.argv)
print (sys.argv)
button = QPushButton("Quit")
button.show()
app.exec()