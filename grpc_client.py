import grpc
from grpc_proto import GEC_pb2, GEC_pb2_grpc

_HOST = '127.0.0.1'
_PORT = '19999'


def main():
    with grpc.insecure_channel("{0}:{1}".format(_HOST, _PORT)) as channel:
        client = GEC_pb2_grpc.ComputeStub(channel=channel)
        text = ["1 2 3 4 5","a b c d e"]
        response = client.GEC(GEC_pb2.GECRequest(text=text))
    print("received: ", response.result)


if __name__ == '__main__':
    main()
