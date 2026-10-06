let io = null;

function initSocket(server) {
  const { Server } = require('socket.io');
  io = new Server(server, {
    cors: {
      origin: '*',
      methods: ['GET', 'POST']
    }
  });

  io.on('connection', (socket) => {
    console.log(`[Socket.io] Client connected: ${socket.id}`);

    socket.on('disconnect', () => {
      console.log(`[Socket.io] Client disconnected: ${socket.id}`);
    });
  });

  return io;
}

function broadcastAlert(alertData) {
  if (io) {
    console.log(`[Socket.io] Broadcasting patient distress alert:`, alertData);
    io.emit('distress_alert', alertData);
  } else {
    console.warn('[Socket.io] io instance not initialized; cannot broadcast alert.');
  }
}

module.exports = {
  initSocket,
  broadcastAlert
};
