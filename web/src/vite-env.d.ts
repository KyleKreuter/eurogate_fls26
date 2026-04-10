/// <reference types="vite/client" />

declare module "*.csv?url" {
  const url: string;
  export default url;
}