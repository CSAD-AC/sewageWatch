const NodeMediaServer = require('node-media-server');
const config = {
  rtmp: { port: 1935, chunk_size: 4096 }, // 接收 RTMP
  http: { port: 8000, allow_origin: '*' }  // 输出 WebSocket
};
const nms = new NodeMediaServer(config);
nms.run();