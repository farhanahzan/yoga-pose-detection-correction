from machine import Pin
import  time

led1 = Pin(16,Pin, OUT)

while True:
    led1.on()
    time.sleep(1)
    led1.off()
    time.sleep(1)