#ifndef UNIFORM_DISPLAY_CONTROL_GLSL
#define UNIFORM_DISPLAY_CONTROL_GLSL

/* META GLOBAL
   @meta: category=Uniform Display Control;
*/

// 显示为三个独立的浮点数滑块
uniform vec3 u_tint_color; /* @subtype=XYZ; */

// 或者更详细地控制每个分量
uniform vec3 u_vector_params; /* @subtype=XYZ; @label=Vector Parameters; */

// 对比：默认显示为颜色
uniform vec3 u_color_param; /* @subtype=COLOR; */

/* META
   @Base_Color: label=Base Color; default=vec3(1.0);
*/
void Test_Vector_Uniform(
    vec3 Base_Color,
    out vec3 Result_Color
)
{
    Result_Color = Base_Color * u_tint_color;
}

/* META
   @Input_Value: label=Input Value; default=1.0;
*/
void Use_Vector_Params(
    float Input_Value,
    out vec3 Result
)
{
    Result = u_vector_params * Input_Value;
}

#endif