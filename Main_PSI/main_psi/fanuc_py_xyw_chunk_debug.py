# Sascha Laube
from __future__ import annotations

import socket


class FanucError(Exception):
    pass


class Robot:
    def __init__(
        self,
        robot_model: str,
        host: str,
        port: int = 18375,
        ee_DO_type: str | None = None,
        ee_DO_num: int | None = None,
        socket_timeout: int = 60,
    ):
        self.robot_model = robot_model
        self.host = host
        self.port = port
        self.ee_DO_type = ee_DO_type
        self.ee_DO_num = ee_DO_num
        self.sock_buff_sz = 1024
        self.socket_timeout = socket_timeout
        self.comm_sock: socket.socket
        self.SUCCESS_CODE = 0
        self.ERROR_CODE = 1

    def handle_response(self, resp: str, continue_on_error: bool = False) -> dict:
        code_, msg = resp.split(":")
        code = int(code_)

        if code == self.ERROR_CODE and not continue_on_error:
            raise FanucError(msg)
        if code not in (self.SUCCESS_CODE, self.ERROR_CODE):
            raise FanucError(f"Unknown response code: {code} and message: {msg}")

        return {"code": code, "msg": msg, "success": code == self.SUCCESS_CODE}

    def connect(self) -> dict:
        try:
            self.comm_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.comm_sock.settimeout(self.socket_timeout)
            self.comm_sock.connect((self.host, self.port))
            resp = self.comm_sock.recv(self.sock_buff_sz).decode()
            return self.handle_response(resp)
        except Exception as e:
            return {"code": self.ERROR_CODE, "msg": str(e), "success": False}

    def disconnect(self) -> None:
        try:
            self.comm_sock.close()
        except Exception as e:
            print(f"Error disconnecting: {e}")

    def send_cmd(self, cmd: str, continue_on_error: bool = False) -> dict:
        try:
            cmd = cmd.strip() + "\n"
            self.comm_sock.sendall(cmd.encode())
            resp = self.comm_sock.recv(self.sock_buff_sz).decode()
            return self.handle_response(resp=resp, continue_on_error=continue_on_error)
        except Exception as e:
            return {"code": self.ERROR_CODE, "msg": str(e), "success": False}

    def setregister(self, cmd: str, continue_on_error: bool = False) -> dict:
        return self.send_cmd(cmd, continue_on_error=continue_on_error)

    def send_vision_data(
        self,
        vision_data: list[tuple[float, float, float]],
        continue_on_error: bool = False,
    ) -> list[dict]:
        responses = []
        try:
            max_points_per_msg = 12
            formatted_vision_data = [
                (f"{x_pos:05.1f}", f"{y_pos:05.1f}", f"{w_orient:05.1f}")
                for x_pos, y_pos, w_orient in vision_data
            ]
            n_chips_vision = len(formatted_vision_data)
            n_chips_vision_ = f"{n_chips_vision:02}"

            for i in range(0, len(formatted_vision_data), max_points_per_msg):
                chunk = formatted_vision_data[i : i + max_points_per_msg]
                cmd_parts = [
                    "vision",
                    str(i // max_points_per_msg + 1),
                    n_chips_vision_,
                ]
                for data in chunk:
                    cmd_parts.extend(data)
                cmd = ":".join(cmd_parts)
                print(cmd)
                responses.append(
                    self.send_cmd(cmd, continue_on_error=continue_on_error)
                )
        except Exception as e:
            responses.append({"code": self.ERROR_CODE, "msg": str(e), "success": False})

        return responses


if __name__ == "__main__":
    robot = Robot(
        robot_model="Fanuc",
        host="192.168.0.100",
        port=18735,
        ee_DO_type="RDO",
        ee_DO_num=7,
    )
    try:
        robot.connect()
    except:
        print("Roboter Verbindung fehlgeschlagen!")

    vision_data = [
        (20, 20, -2.9),
        (20, 60, 2.9),
        (20, 100, -2.9),
        (20, 140, 2.9),
        (60, 20, -2.9),
        (60, 60, 2.9),
        (60, 100, -2.9),
        (60, 140, 2.9),
        (100, 20, -2.9),
        (100, 60, 2.9),
        (100, 100, -2.9),
        (100, 140, 2.9),
        (140, 20, -2.9),
        (140, 60, 2.9),
        (140, 100, -2.9),
        (140, 140, 2.9),
        (180, 20, -2.9),
        (180, 60, 2.9),
        (180, 100, -2.9),
        (180, 140, 2.9),
        (220, 20, -2.9),
        (220, 60, 2.9),
        (220, 100, -2.9),
        (220, 140, 2.9),
        (260, 20, -2.9),
        (260, 60, 2.9),
        (260, 100, -2.9),
        (260, 140, 2.9),
        (300, 20, -2.9),
        (300, 60, 2.9),
        (300, 100, -2.9),
        (300, 140, 2.9),
        (340, 20, -2.9),
        (340, 60, 2.9),
        (340, 100, -2.9),
        (340, 140, 2.9),
    ]
    robot.send_vision_data(vision_data)
    robot.disconnect()
