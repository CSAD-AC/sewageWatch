const NodeMediaServer = require('node-media-server');
const config = {
  rtmp: { port: 1935, chunk_size: 4096 }, // 接收 RTMP
};
const nms = new NodeMediaServer(config);
nms.run();