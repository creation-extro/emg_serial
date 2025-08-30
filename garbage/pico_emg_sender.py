# Raspberry Pi Pico Code for Real-Time EMG Data Transmission
# Upload this to your Pico using Thonny
# Connect EMG sensors to GPIO 26, 27, 28

import machine
import time
import utime

# ADC pins for EMG channels
adc0 = machine.ADC(26)  # CH1 - EMG Channel 1 (e.g., bicep)
adc1 = machine.ADC(27)  # CH2 - EMG Channel 2 (e.g., forearm)
adc2 = machine.ADC(28)  # CH3 - EMG Channel 3 (e.g., wrist)

# LED for status indication
led = machine.Pin(25, machine.Pin.OUT)

# Built-in LED for connection status
onboard_led = machine.Pin("LED", machine.Pin.OUT)

def read_emg_channels():
    """Read all 3 EMG channels with filtering"""
    # Read multiple samples and average for noise reduction
    samples = 5
    ch1_sum = ch2_sum = ch3_sum = 0

    for _ in range(samples):
        ch1_sum += adc0.read_u16()
        ch2_sum += adc1.read_u16()
        ch3_sum += adc2.read_u16()
        time.sleep_us(100)  # Small delay between samples

    ch1 = ch1_sum // samples
    ch2 = ch2_sum // samples
    ch3 = ch3_sum // samples

    return ch1, ch2, ch3

def calibrate_baseline():
    """Calibrate baseline EMG values (muscle at rest)"""
    print("🔧 Calibrating baseline...")
    print("💪 Keep muscles RELAXED for 3 seconds...")

    baseline_samples = []

    for i in range(300):  # 3 seconds at 100Hz
        ch1, ch2, ch3 = read_emg_channels()
        baseline_samples.append([ch1, ch2, ch3])

        if i % 100 == 0:
            print(f"   {3 - i//100} seconds remaining...")

        time.sleep(0.01)

    # Calculate baseline averages
    baseline = [0, 0, 0]
    for sample in baseline_samples:
        for i in range(3):
            baseline[i] += sample[i]

    for i in range(3):
        baseline[i] //= len(baseline_samples)

    print(f"✅ Baseline calibrated: {baseline}")
    return baseline

def main():
    """Main EMG data transmission loop"""
    print("🎮 Real-Time EMG Data Sender")
    print("📡 Sending muscle signals to PC...")
    print("🔌 Connect to AR + LightGBM model")
    print("=" * 40)

    # Turn on onboard LED to show system is ready
    onboard_led.on()

    # Calibrate baseline
    baseline = calibrate_baseline()

    print("\n🚀 Starting real-time EMG transmission...")
    print("💪 Perform your hand gestures!")
    print("🛑 Press Ctrl+C to stop")
    print("-" * 40)

    sample_count = 0

    try:
        while True:
            # Read EMG channels
            ch1, ch2, ch3 = read_emg_channels()

            # Apply baseline correction
            ch1_corrected = max(0, ch1 - baseline[0])
            ch2_corrected = max(0, ch2 - baseline[1])
            ch3_corrected = max(0, ch3 - baseline[2])

            # Send data in CSV format with timestamp
            timestamp = time.ticks_ms()
            print(f"{timestamp},{ch1_corrected},{ch2_corrected},{ch3_corrected}")

            # Blink LED to show activity
            led.toggle()

            sample_count += 1

            # Status update every 1000 samples
            if sample_count % 1000 == 0:
                print(f"# Sent {sample_count} samples")

            # Stable sampling rate (~100Hz)
            time.sleep(0.01)  # 10ms = 100Hz

    except KeyboardInterrupt:
        print(f"\n🛑 EMG transmission stopped")
        print(f"📊 Total samples sent: {sample_count}")
        led.off()
        onboard_led.off()

    except Exception as e:
        print(f"❌ Error: {e}")
        led.off()
        onboard_led.off()
        time.sleep(0.1)

if __name__ == "__main__":
    main()
