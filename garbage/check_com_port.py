#!/usr/bin/env python3
"""
Check COM Port Availability
Test if COM port is available and can be opened
"""

import serial
import serial.tools.list_ports
import time

def check_com_ports():
    """Check all available COM ports"""
    print("🔍 Checking available COM ports...")
    
    ports = serial.tools.list_ports.comports()
    
    if not ports:
        print("❌ No COM ports found")
        return []
    
    print(f"📊 Found {len(ports)} COM ports:")
    available_ports = []
    
    for port in ports:
        print(f"\n📍 {port.device}:")
        print(f"   Description: {port.description}")
        print(f"   Hardware ID: {port.hwid}")
        
        # Try to open the port
        try:
            ser = serial.Serial(port.device, 115200, timeout=1)
            ser.close()
            print(f"   ✅ Available")
            available_ports.append(port.device)
        except serial.SerialException as e:
            print(f"   ❌ Busy/Error: {e}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    return available_ports

def test_specific_port(port_name):
    """Test a specific COM port"""
    print(f"\n🔍 Testing {port_name}...")
    
    try:
        # Try to open port
        ser = serial.Serial(port_name, 115200, timeout=2)
        print(f"✅ {port_name} opened successfully")
        
        # Try to read some data
        print("📊 Reading data for 5 seconds...")
        start_time = time.time()
        data_count = 0
        
        while time.time() - start_time < 5:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    print(f"   Data: {line}")
                    data_count += 1
                    if data_count >= 5:  # Show first 5 lines
                        break
            time.sleep(0.1)
        
        ser.close()
        
        if data_count > 0:
            print(f"✅ {port_name} is working and sending data!")
            return True
        else:
            print(f"⚠️  {port_name} opened but no data received")
            return False
            
    except serial.SerialException as e:
        print(f"❌ Cannot open {port_name}: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing {port_name}: {e}")
        return False

def main():
    """Main function"""
    print("🔧 COM Port Checker")
    print("=" * 40)
    
    # Check all ports
    available_ports = check_com_ports()
    
    if not available_ports:
        print("\n❌ No available COM ports found")
        print("🔧 Possible solutions:")
        print("   1. Close Thonny or other serial programs")
        print("   2. Unplug and replug your Pico")
        print("   3. Restart your computer")
        return
    
    print(f"\n✅ Available ports: {available_ports}")
    
    # Test specific port
    if 'COM3' in available_ports:
        print(f"\n🎯 Testing COM3 (your Pico port)...")
        if test_specific_port('COM3'):
            print(f"\n🎉 COM3 is ready for deep learning prediction!")
            print(f"🚀 You can now run: python deep_learning_realtime_tester.py")
        else:
            print(f"\n⚠️  COM3 has issues. Try:")
            print(f"   1. Restart Pico code in Thonny")
            print(f"   2. Check EMG sensor connections")
    else:
        print(f"\n❌ COM3 not available")
        print(f"🔧 Try these ports instead: {available_ports}")

if __name__ == "__main__":
    main()
