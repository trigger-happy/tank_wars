// Shader taken from the PostProcessing example in ClanLib 2.2.3

varying vec4 Color;
varying vec2 TexCoord;

uniform sampler2D Texture;
uniform float Amount;
uniform float Timer;

void main()
{
	vec4 sum = vec4(0);
	vec2 texcoord = TexCoord;
	int j;
	int i;
	
	for( i= -4 ;i < 4; i++)
	{
		for (j = -3; j < 3; j++)
		{
			sum += texture2D(Texture, texcoord + vec2(j, i)*0.004) * 0.25;
		}
	}
	if (texture2D(Texture, texcoord).r < 0.3)
	{
		gl_FragColor = sum*sum*0.012 + texture2D(Texture, texcoord);
	}
	else
	{
		if (texture2D(Texture, texcoord).r < 0.5)
		{
			gl_FragColor = sum*sum*0.009 + texture2D(Texture, texcoord);
		}
		else
		{
			gl_FragColor = sum*sum*0.0075 + texture2D(Texture, texcoord);
		}
	}
}