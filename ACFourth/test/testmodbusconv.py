import struct
from pymodbus.client import ModbusTcpClient
import struct
from pymodbus.client import ModbusTcpClient

# กำหนดค่าการเชื่อมต่อ
SERVER_IP = '192.168.1.201'
PORT = 502
SLAVE_ID = 1

# ตำแหน่ง Register ที่ต้องการอ่าน
REGISTER_MAP = {
    "V": 1020,
    "I": 1000,
    "P": 1034,
    "Q": 1042,
}

BAUDRATE_REG = 81
BAUDRATE_MAP = {
    "1": 4800,
    "2": 9600,
    "3": 19200,
    "4": 38400,
    "5": 57600,
    "6": 115200,
}

ct_reg = 505

def decode_float32(registers):
    """
    รับค่าลิสต์ของ Register 2 ตัว แล้ว Decode กลับเป็น Float 32-bit
    """
    if len(registers) != 2:
        return None
    
    # Pack 16-bit int 2 ตัว (H = Unsigned Short) แบบ Big-Endian (>)
    # รูปแบบ ABCD (Standard)
    packed_bytes = struct.pack('>HH', registers[0], registers[1])
    
    # *หมายเหตุ: ถ้ารันแล้วค่า Float ออกมาแปลกๆ หรือผิดเพี้ยน 
    # แสดงว่ามิเตอร์อาจจะส่งข้อมูลแบบ Word-Swap (CDAB) ให้ปิดบรรทัดบนและเปิดใช้บรรทัดล่างแทน
    # packed_bytes = struct.pack('>HH', registers[1], registers[0])
    
    # Unpack 4 bytes กลับมาเป็น Float 32-bit (f)
    float_val = struct.unpack('>f', packed_bytes)[0]
    return float_val

def decode_uint16(registers):
    """
    รับค่าลิสต์ของ Register 1 ตัว แล้ว Decode กลับเป็น Unsigned Int 16-bit
    """
    if len(registers) != 1:
        return None
    
    # Pack 16-bit int 1 ตัว (H = Unsigned Short) แบบ Big-Endian (>)
    packed_bytes = struct.pack('>H', registers[0])
    
    # Unpack กลับมาเป็น Unsigned Int 16-bit (H)
    uint16_val = struct.unpack('>H', packed_bytes)[0]
    return uint16_val

def decode_uint32(registers):
    """
    รับค่าลิสต์ของ Register 2 ตัว แล้ว Decode กลับเป็น Unsigned Int 32-bit
    """
    if len(registers) != 2:
        return None
    
    # Pack 16-bit int 2 ตัว (H = Unsigned Short) แบบ Big-Endian (>)
    packed_bytes = struct.pack('>HH', registers[0], registers[1])
    
    # Unpack กลับมาเป็น Unsigned Int 32-bit (I)
    uint32_val = struct.unpack('>I', packed_bytes)[0]
    return uint32_val

def main():
    client = ModbusTcpClient(SERVER_IP, port=PORT)
    
    if client.connect():
        print(f"Connected to {SERVER_IP}:{PORT} successfully.\n")
        print("-" * 30)
        
        # 1. อ่านค่า Baudrate (uint 1 register)
        result_baud = client.read_holding_registers(address=BAUDRATE_REG, count=1, slave=SLAVE_ID)
        if not result_baud.isError():
            baudrate_index = decode_uint16(result_baud.registers)
            
            # แมปค่า Index (ที่แปลงเป็น string) เข้ากับ BAUDRATE_MAP
            # ใช้ .get() เพื่อป้องกัน Error กรณีที่อ่านได้ค่าขยะที่ไม่อยู่ใน Map
            actual_baudrate = BAUDRATE_MAP.get(str(baudrate_index), f"Unknown Index ({baudrate_index})")
            
            print(f"Baudrate (Reg {BAUDRATE_REG}): {actual_baudrate} bps")
        else:
            print(f"Error reading Baudrate (Reg {BAUDRATE_REG})")
            
        print("-" * 30)

        result_ct = client.read_holding_registers(address=ct_reg, count=2, slave=SLAVE_ID)
        if not result_baud.isError():
            print(result_ct.registers)
            ct_index = decode_uint32(result_ct.registers)
            
            # แมปค่า Index (ที่แปลงเป็น string) เข้ากับ BAUDRATE_MAP
            # ใช้ .get() เพื่อป้องกัน Error กรณีที่อ่านได้ค่าขยะที่ไม่อยู่ใน Map
            
            print(f"Baudrate (Reg {ct_reg}): {ct_index}")
        else:
            print(f"Error reading Baudrate (Reg {ct_reg})")
            
        print("-" * 30)
        
        # 2. วนลูปอ่านค่า V, I, P, Q (Float 2 registers)
        for name, addr in REGISTER_MAP.items():
            # อ่านทีละ 2 Registers สำหรับค่า Float
            result = client.read_holding_registers(address=addr, count=2, slave=SLAVE_ID)
            
            if not result.isError():
                regs = result.registers
                value = decode_float32(regs)
                # ฟอร์แมตให้แสดงทศนิยม 2 ตำแหน่ง
                print(f"{name} (Reg {addr}): {value:.6f}")
            else:
                print(f"Error reading {name} (Reg {addr})")
                
        print("-" * 30)
        client.close()
    else:
        print(f"Failed to connect to {SERVER_IP}:{PORT}")

if __name__ == "__main__":
    main()