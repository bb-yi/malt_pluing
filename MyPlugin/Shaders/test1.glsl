#include "Common.glsl"

void starting_point()
{

}

void flat_color(vec3 color, out vec3 result)
{
    result = color;
}

void textures(int uv_channel, sampler2D color_texture, out vec4 result)
{
    vec2 texture_coordinates = UV[uv_channel];

    vec4 sampled_color = texture(color_texture, texture_coordinates);

    result = sampled_color;
}