#define PI 3.1415926535897932384626433832795
/* META
    @label: Smoothstep;
    @edge0: subtype=Slider;default=0.0;min=0.0;max=1.0;
    @edge1: subtype=Slider;default=1.0;min=0.0;max=1.0;
*/
void my_smoothstep( in float x,in float edge0, in float edge1, out float result) {
    result = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    result = result * result * (3.0 - 2.0 * result);
}
//三种颜色的渐变
/*  META
    @position1: subtype=Slider;default=0.0;min=0.0;max=1.0;
    @position2: subtype=Slider;default=0.5;min=0.0;max=1.0;
    @position3: subtype=Slider;default=1.0;min=0.0;max=1.0;
    @color1: default=(0.0, 0.0, 0.0);
    @color2: default=(0.5, 0.5, 0.5);
    @color3: default=(1.0, 1.0, 1.0);
*/
void ramp_3colors(in float x,in float position1,in vec3 color1, 
                    in float position2,in vec3 color2, 
                    in float position3,in vec3 color3,
                    out vec3 result)
{
    if (x <= position1) {
        result = color1;
    }
    else if (x >= position3) {
        result = color3;
    }
    else if (x <= position2) {
        // 在 [position1, position2] 区间进行 color1 → color2 插值
        float t = (x - position1) / (position2 - position1);
        result = mix(color1, color2, t);
    }
    else {
        // 在 [position2, position3] 区间进行 color2 → color3 插值
        float t = (x - position2) / (position3 - position2);
        result = mix(color2, color3, t);
    }
}
/*  META
    @UV: label=UV; default=UV[0];
    @smoothness: subtype=Slider;default=0.02;min=0.0;max=1.0;
*/
void toon_sdf_mask(in sampler2D SDF, in vec2 UV,in float angle,in float smoothness,in float offset, out float mask,out float SDF_R,out float SDF_G,out float SDF_B,out float SDF_A)
{
    vec4 SDF_RIGHT = texture(SDF, UV);
    float SDF_R_F =( SDF_RIGHT.r+SDF_RIGHT.b+SDF_RIGHT.a)/3.0+offset;
    vec4 SDF_LEFT = texture(SDF, vec2(1.0-UV.x, UV.y));
    float SDF_L_F =(SDF_LEFT.r+SDF_LEFT.b+SDF_LEFT.a)/3.0+offset;
    angle+=PI;
    angle =mod(angle+PI, 2.0*PI)-PI;
    float Threshold = -cos(angle)*0.5+0.5;
    if (angle>0.0)
    {
    mask = smoothstep(Threshold-smoothness,Threshold+smoothness,SDF_R_F);
    }
    else
    {
    mask = smoothstep(Threshold-smoothness,Threshold+smoothness,SDF_L_F);
    }
    SDF_R = SDF_RIGHT.r;
    SDF_G = SDF_RIGHT.g;
    SDF_B = SDF_RIGHT.b;
    SDF_A = SDF_RIGHT.a;
}

//采样绿色法线
/*  META
    @UV: label=UV; default=UV[0];
*/
void Smapler_green_normal(in sampler2D tex, in vec2 UV, out vec3 normal,out float tex_b,out float tex_a)
{
    vec4 tex_color = texture(tex, UV);
    normal = vec3(tex_color.r, tex_color.g, sqrt(1-pow(tex_color.r, 2)-pow(tex_color.g, 2)));
    tex_b = tex_color.b;
    tex_a = tex_color.a;
}

//法线缩放
/*  META
    @baseNormal: default=NORMAL;
*/
void NormalScale(in vec3 normal,in vec3 baseNormal,in float scale, out vec3 result,out vec3 baseNormal_out)
{
    result = normalize(normal * scale + baseNormal * (1.0 - scale));
    baseNormal_out = baseNormal;
}

void GetImageSize(in sampler2D Image, out vec2 size)
{
    size = vec2(textureSize(Image, 0));
}
void HasImage(in sampler2D Image, out bool result)
{
    result = textureSize(Image, 0).x > 1;
}
/* META @meta: label=My HSV Edit; */
vec4 my_hsv_edit(vec4 color, float mask,float hue, float saturation, float value)
{
    vec3 hsv = rgb_to_hsv(color.rgb);
    hsv += vec3(hue, saturation, value)*vec3(mask);
    return vec4(hsv_to_rgb(hsv), color.a);
}


/* META
    @x: default=0;
*/
vec3 packIntToVec3(int x) {
    float fx = float(x);
    vec3 v;
    v.x = mod(fx, 256.0);     fx = floor(fx / 256.0);
    v.y = mod(fx, 256.0);     fx = floor(fx / 256.0);
    v.z = mod(fx, 256.0);

    return v / 255.0;  // 归一化到 [0,1]
}


/* META
    @v: default=(0.0, 0.0,0.0,0.0);
*/
int unpackVec3ToInt(vec3 v) {
    vec3 bytes = round(v * 255.0);  // 还原 0~255
    float value = bytes.x +
                  bytes.y * 256.0 +
                  bytes.z * 65536.0;
    return int(value);
}

/* META
    @Position: subtype=Vector; default=(0,0,0,0);
    @Projection: subtype=Matrix; default=PROJECTION;
    @depth: default=0.0;
    @normalizedDepth: default=0.0;
    @isBackground: default=false;
    @isForeground: default=ture;
*/
/*  META
    @Position      : subtype=Vector; default=(0,0,0,0);
    @Projection    : subtype=Matrix; default=PROJECTION;
    @depth         : default=0.0;
    @normalizedDepth: default=0.0;
    @isBackground  : default=false;
    @isForeground  : default=true;
*/
void PositionToDepthsProj(
    in  vec4 Position,           // 世界空间位置（w==0 表示背景）
    in  mat4 Projection,         // 相机投影矩阵
    out float depth,             // 相机空间线性深度（世界单位）
    out float normalizedDepth,   // 线性归一化深度 [0,1]
    out bool  isBackground,      // 背景标记
    out bool  isForeground       // 前景标记
){
    /* ---------- 1. 背景快速分支 ---------- */
    if (Position.w == 0.0)
    {
        depth           = 1000.0;   // 足够大的线性深度
        normalizedDepth = 1.0;
        isBackground    = true;
        isForeground    = false;
        return;
    }

    /* ---------- 2. 世界 → 相机空间 ---------- */
    vec3 worldPos = Position.xyz;
    Transform(0,      // Type = Point
              1,      // From = World
              2,      // To   = Camera
              worldPos);
    vec3 viewPos = worldPos;   // 此时已在相机空间

    /* ---------- 3. 线性深度（OpenGL 相机朝 -Z） ---------- */
    depth = -viewPos.z;

    /* ---------- 4. 线性归一化深度 [0,1] ---------- */
    float n = abs(Projection[3][2] / (Projection[2][2] - 1.0)); // 近平面
    float f = abs(Projection[3][2] / (Projection[2][2] + 1.0)); // 远平面
    normalizedDepth = clamp((depth - n) / (f - n), 0.0, 1.0);

    /* ---------- 5. 前景/背景标记 ---------- */
    isBackground = false;
    isForeground = true;
}