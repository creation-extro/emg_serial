#!/usr/bin/env python3
"""
Find Available COM Ports
Detect which COM ports are available for Pico connection
"""

import serial.tools.list_ports
import serial
import time

def list_com_ports():
    """List all available COM ports"""
    print("🔍 Scanning for COM ports...")
    ports = serial.tools.list_ports.comports()
    
    if not ports:
        print("❌ No COM ports found!")
        return []
    
    print(f"✅ Found {len(ports)} COM port(s):")
    available_ports = []
    
    for port in ports:
        print(f"\n📡 {port.device}:")
        print(f"   Description: {port.description}")
        print(f"   Hardware ID: {port.hwid}")
        
        # Check if it's likely a Pico
        if any(keyword in port.description.lower() for keyword in ['usb', 'serial', 'pico', 'rp2040']):
            print(f"   🎯 Likely Pico device!")
            available_ports.append(port.device)
        
        # Check if port is accessible
        try:
            test_serial = serial.Serial(port.device, 115200, timeout=1)
            test_serial.close()
            print(f"   ✅ Port accessible")
        except serial.SerialException as e:
            if "Access is denied" in str(e):
                print(f"   ⚠️  Port busy (probably used by Thonny)")
            else:
                print(f"   ❌ Port error: {e}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    return available_ports

def test_pico_connection(port):
    """Test if Pico is responding on this port"""
    print(f"\n🧪 Testing Pico connection on {port}...")
    
    try:
        ser = serial.Serial(port, 115200, timeout=2)
        time.sleep(1)
        
        print("📡 Listening for Pico data...")
        
        # Try to read some data
        for i in range(10):
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    print(f"   📊 Received: {line}")
                    
                    # Check if it looks like EMG data
                    if ',' in line and not line.startswith('#'):
                        parts = line.split(',')
                        if len(parts) >= 3:
                            try:
                                # Try to parse as numbers
                                [int(x) for x in parts[:3]]
                                print(f"   ✅ Looks like EMG data!")
                                ser.close()
                                return True
                            except ValueError:
                                pass
            
            time.sleep(0.1)
        
        ser.close()
        print(f"   ⚠️  No EMG data detected")
        return False
        
    except serial.SerialException as e:
        if "Access is denied" in str(e):
            print(f"   ⚠️  Port busy - close Thonny first!")
        else:
            print(f"   ❌ Connection failed: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def main():
    """Main function"""
    print("🔍 COM Port Detector for Pico")
    print("=" * 40)
    
    # List all ports
    available_ports = list_com_ports()
    
    if not available_ports:
        print("\n❌ No suitable ports found!")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure Pico is connected via USB")
        print("2. Check if Pico appears in Device Manager")
        print("3. Try a different USB cable")
        print("4. Try a different USB port")
        return
    
    print(f"\n🎯 Recommended ports for testing: {available_ports}")
    
    # Test each port
    print(f"\n🧪 Testing ports for Pico EMG data...")
    print("⚠️  Make sure to CLOSE THONNY first!")
    
    input("Press Enter when Thonny is closed...")
    
    working_ports = []
    
    for port in available_ports:
        if test_pico_connection(port):
            working_ports.append(port)
    
    if working_ports:
        print(f"\n✅ Working Pico ports: {working_ports}")
        print(f"🎯 Use this port: {working_ports[0]}")
    else:
        print(f"\n⚠️  No ports with EMG data found")
        print(f"🔧 Make sure:")
        print(f"   1. Pico code is running (check Thonny)")
        print(f"   2. Thonny is closed")
        print(f"   3. EMG sensors are connected")

if __name__ == "__main__":
    main()
