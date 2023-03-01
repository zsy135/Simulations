import pickle


class PacketType:
    UnValid_P = 0
    ModelReq_P = 1
    ModelWei_P = 2
    ModelBroadCast_P = 3
    ConnectReq_P = 4


class Packet:
    def __init__(self, data=None, type=0):
        self.data = data
        self.type = type

    def decode(self):
        if self.data is not None:
            self.type = int.from_bytes(self.data[:1], "little")
            if self.type == PacketType.ModelReq_P:
                return WeiReqPack.decode_from_data(self.data)

            elif self.type == PacketType.ModelWei_P:
                return ModelWeiPack.decode_from_data(self.data)


            elif self.type == PacketType.ModelBroadCast_P:
                return ModelBroadCastPack.decode_from_data(self.data)

            elif self.type == PacketType.ConnectReq_P:
                return ConnectReqPack.decode_from_data(self.data)
            else:
                return None

    @classmethod
    def decode_from_data(cls, data):
        pass


class WeiReqPack(Packet):
    def __init__(self):
        super(WeiReqPack, self).__init__(None, PacketType.ModelReq_P)

    @classmethod
    def decode_from_data(cls, data):
        p = WeiReqPack()
        p.data = data
        return p

    def encode(self):
        self.data = self.type.to_bytes(1, 'little')
        return self.data

class ConnectReqPack(Packet):
    def __init__(self,id_=None):
        super(ConnectReqPack, self).__init__(None, PacketType.ConnectReq_P)
        self.id = id_

    @classmethod
    def decode_from_data(cls, data):
        p = ConnectReqPack()
        p.data = data
        pInfo = pickle.loads(data[1:])
        p.id = pInfo['id']
        return p

    def encode(self):
        self.data = self.type.to_bytes(1, 'little')+pickle.dumps({"id": self.id})
        return self.data




class ModelWeiPack(Packet):

    def __init__(self, weight=None, acc=None, version=None):
        super(ModelWeiPack, self).__init__(None, PacketType.ModelWei_P)
        self.weight = weight
        self.acc = acc
        self.version = version

    @classmethod
    def decode_from_data(cls, data):
        p = ModelWeiPack()
        p.data = data
        pInfo = pickle.loads(data[1:])
        p.weight = pInfo['w']
        p.acc = pInfo['a']
        p.version = pInfo['v']
        return p

    def encode(self):
        self.data = self.type.to_bytes(1, 'little') + pickle.dumps({"w": self.weight, 'a': self.acc, 'v': self.version})
        return self.data


class ModelBroadCastPack(Packet):

    def __init__(self, weight=None, acc=None, version=None,owner=None):
        super(ModelBroadCastPack, self).__init__(None, PacketType.ModelBroadCast_P)
        self.weight = weight
        self.acc = acc
        self.version = version
        self.owner = owner

    @classmethod
    def decode_from_data(cls, data):
        p = ModelBroadCastPack()
        p.data = data
        pInfo = pickle.loads(data[1:])
        p.weight = pInfo['w']
        p.acc = pInfo['a']
        p.version = pInfo['v']
        p.owner = pInfo['o']
        return p

    def encode(self):
        self.data = self.type.to_bytes(1, 'little') + pickle.dumps({"w": self.weight, 'a': self.acc, 'v': self.version,'o':self.owner})
        return self.data