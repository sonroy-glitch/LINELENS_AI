# LED + buzzer alerts for raspberry pi
# connects to GPIO pins to light up LEDs and buzz a piezo buzzer
# green = all good, yellow = warning, red + buzz = danger
#
# WIRING:
# green LED  -> GPIO 17 (pin 11)
# yellow LED -> GPIO 27 (pin 13)
# red LED    -> GPIO 22 (pin 15)
# buzzer     -> GPIO 18 (pin 12)
# all grounds to GND pins
#
# if you're not on a raspi this script will just print to console instead

import time

# try to import GPIO, if it fails we're not on a pi
try:
    import RPi.GPIO as GPIO
    ON_PI = True
except ImportError:
    ON_PI = False
    print("Not on Raspberry Pi - running in simulation mode")

GREEN = 17
YELLOW = 27
RED = 22
BUZZER = 18

if ON_PI:
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    for pin in [GREEN, YELLOW, RED, BUZZER]:
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, GPIO.LOW)

def set_leds(g, y, r, buzz):
    """set led and buzzer states. True = on, False = off"""
    if ON_PI:
        GPIO.output(GREEN, g)
        GPIO.output(YELLOW, y)
        GPIO.output(RED, r)
        GPIO.output(BUZZER, buzz)
    else:
        status = []
        if g: status.append("GREEN")
        if y: status.append("YELLOW")
        if r: status.append("RED")
        if buzz: status.append("BUZZER")
        print(f"  LEDs: {', '.join(status) if status else 'all off'}")

def alert_normal():
    """green light - everything is fine"""
    set_leds(True, False, False, False)

def alert_warning():
    """yellow blink - idle too long, SOP drift, etc"""
    set_leds(False, True, False, False)
    time.sleep(0.5)
    set_leds(False, False, False, False)
    time.sleep(0.5)

def alert_danger():
    """red + buzzer - hazard zone, near miss"""
    set_leds(False, False, True, True)
    time.sleep(0.3)
    set_leds(False, False, False, False)
    time.sleep(0.2)

def alert_ergo():
    """yellow pulse - ergonomic violation"""
    for _ in range(3):
        set_leds(False, True, False, False)
        time.sleep(0.2)
        set_leds(False, False, False, False)
        time.sleep(0.2)

def cleanup():
    if ON_PI:
        for pin in [GREEN, YELLOW, RED, BUZZER]:
            GPIO.output(pin, False)
        GPIO.cleanup()

# demo - cycle through all patterns
if __name__ == "__main__":
    print("LED alert demo. Ctrl+C to stop.")
    try:
        while True:
            print("\n--- NORMAL ---")
            alert_normal()
            time.sleep(2)

            print("\n--- WARNING ---")
            for _ in range(3):
                alert_warning()

            print("\n--- DANGER ---")
            for _ in range(3):
                alert_danger()

            print("\n--- ERGO ---")
            alert_ergo()
    except KeyboardInterrupt:
        pass
    finally:
        cleanup()
        print("\nDone.")
