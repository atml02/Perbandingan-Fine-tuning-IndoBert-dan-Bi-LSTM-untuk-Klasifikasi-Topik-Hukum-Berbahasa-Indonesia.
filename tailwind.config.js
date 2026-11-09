/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./templates/*"],
  theme: {
    extend: {
      colors:{
        "primary": "#070F2B",
        "secondary": "#8E7AB5", 
        "tartiary": "#999999", 
        "pink": "#EE9AE5",
        "warning": "#D0A2F7",
        "light-purple": "#E3DFFD",
      },
      boxShadow: {
        '3xl': '0px 10px 50px 0px rgba(0, 0, 0, 0.15)'
      },
      backgroundImage: {
        radialBgCircle: "url('/src/assets/img/radial-bg.svg')",
        videoBg: "url('/static/images/img/video-bg.png')",
        circle: "url('/src/assets/img/circle.png')",
        explore: "url('/src/assets/img/explore.png')",
      },
    },
  },
  plugins: [],
}