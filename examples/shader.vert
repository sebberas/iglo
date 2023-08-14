#version 450

layout(location = 0) in vec2 a_position;
layout(location = 1) in vec4 a_color;

layout(location = 0) out vec4 color;

layout(binding = 0) uniform Matrices {
  mat4 modelMatrix;
  mat4 viewMatrix;
  mat4 projectionMatrix;
};

void main() {
  gl_Position = vec4(a_position, 0.0, 1.0);
  color = a_color;
}