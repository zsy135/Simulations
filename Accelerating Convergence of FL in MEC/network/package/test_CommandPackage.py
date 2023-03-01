from CommandPackage import CmdPack

p = CmdPack(4, 1125)
p.print()
en_data = p.pack()
de_data = p.unpack(en_data)
print(en_data.hex())
print(de_data)
